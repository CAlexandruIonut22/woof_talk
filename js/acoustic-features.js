/**
 * Research-backed acoustic features for canine vocalizations.
 * Mirrors ml/feature_extraction.py for browser inference.
 *
 * References:
 * - Yin & McCowan (2004), Animal Behaviour — context-specific bark acoustics
 * - Molnár et al. (2008), Applied Animal Behaviour Science — ML bark contexts
 * - Pongrácz et al. (2005), Applied Animal Behaviour Science — human perception
 */

export const FEATURE_NAMES = [
  "log_duration",
  "rms",
  "spectral_centroid",
  "spectral_rolloff",
  "zero_crossing_rate",
  "low_band_ratio",
  "mid_band_ratio",
  "high_band_ratio",
  "f0_mean",
  "f0_std",
  "harmonicity",
  "bark_count",
  "mean_inter_bark_ms",
  "repetition_rate_hz",
];

function bandRatios(freqData, sampleRate, fftSize) {
  const binWidth = sampleRate / fftSize;
  let low = 0;
  let mid = 0;
  let high = 0;

  for (let i = 0; i < freqData.length; i += 1) {
    const freq = i * binWidth;
    const value = freqData[i];
    if (freq < 500) low += value;
    else if (freq < 2000) mid += value;
    else high += value;
  }

  const total = low + mid + high || 1;
  return { low: low / total, mid: mid / total, high: high / total };
}

function estimateF0(channel, sampleRate) {
  const minLag = Math.floor(sampleRate / 800);
  const maxLag = Math.floor(sampleRate / 80);
  const windowSize = Math.min(2048, channel.length);
  const start = Math.floor((channel.length - windowSize) / 2);
  const window = channel.slice(start, start + windowSize);

  let bestLag = 0;
  let bestCorr = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let corr = 0;
    let normA = 0;
    let normB = 0;
    const limit = windowSize - lag;
    for (let i = 0; i < limit; i += 1) {
      const a = window[i];
      const b = window[i + lag];
      corr += a * b;
      normA += a * a;
      normB += b * b;
    }
    const score = corr / (Math.sqrt(normA * normB) + 1e-9);
    if (score > bestCorr) {
      bestCorr = score;
      bestLag = lag;
    }
  }

  if (bestLag === 0 || bestCorr < 0.2) {
    return { mean: 0, std: 0 };
  }

  const f0 = sampleRate / bestLag;
  const segmentCount = 6;
  const segmentLen = Math.floor(windowSize / segmentCount);
  const f0Values = [];

  for (let s = 0; s < segmentCount; s += 1) {
    const seg = window.slice(s * segmentLen, (s + 1) * segmentLen);
    let segBestLag = 0;
    let segBestCorr = -Infinity;
    for (let lag = minLag; lag <= maxLag; lag += 1) {
      let corr = 0;
      let normA = 0;
      let normB = 0;
      const limit = seg.length - lag;
      for (let i = 0; i < limit; i += 1) {
        corr += seg[i] * seg[i + lag];
        normA += seg[i] ** 2;
        normB += seg[i + lag] ** 2;
      }
      const score = corr / (Math.sqrt(normA * normB) + 1e-9);
      if (score > segBestCorr) {
        segBestCorr = score;
        segBestLag = lag;
      }
    }
    if (segBestLag > 0 && segBestCorr >= 0.2) {
      f0Values.push(sampleRate / segBestLag);
    }
  }

  if (f0Values.length === 0) {
    return { mean: f0, std: 0 };
  }

  const mean = f0Values.reduce((a, b) => a + b, 0) / f0Values.length;
  const variance =
    f0Values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / f0Values.length;
  return { mean, std: Math.sqrt(variance) };
}

function harmonicity(channel) {
  let harmonicEnergy = 0;
  let percussiveEnergy = 0;
  for (let i = 2; i < channel.length - 2; i += 1) {
    const smooth = (channel[i - 1] + channel[i] + channel[i + 1]) / 3;
    const rough = channel[i] - smooth;
    harmonicEnergy += smooth ** 2;
    percussiveEnergy += rough ** 2;
  }
  return harmonicEnergy / (harmonicEnergy + percussiveEnergy + 1e-9);
}

function detectBarkEvents(channel, sampleRate) {
  const envelope = new Float32Array(channel.length);
  for (let i = 0; i < channel.length; i += 1) {
    envelope[i] = Math.abs(channel[i]);
  }

  let max = 0;
  for (let i = 0; i < envelope.length; i += 1) {
    max = Math.max(max, envelope[i]);
  }
  const threshold = max * 0.35;
  const minGap = Math.floor(sampleRate * 0.08);
  const peaks = [];
  let lastPeak = -minGap;

  for (let i = 1; i < envelope.length - 1; i += 1) {
    if (
      envelope[i] > threshold &&
      envelope[i] >= envelope[i - 1] &&
      envelope[i] >= envelope[i + 1] &&
      i - lastPeak >= minGap
    ) {
      peaks.push(i);
      lastPeak = i;
    }
  }

  if (peaks.length < 2) {
    return { barkCount: peaks.length, meanInterBarkMs: 0, repetitionRateHz: 0 };
  }

  const intervals = [];
  for (let i = 1; i < peaks.length; i += 1) {
    intervals.push(((peaks[i] - peaks[i - 1]) / sampleRate) * 1000);
  }
  const meanInterBarkMs = intervals.reduce((a, b) => a + b, 0) / intervals.length;
  return {
    barkCount: peaks.length,
    meanInterBarkMs,
    repetitionRateHz: meanInterBarkMs > 0 ? 1000 / meanInterBarkMs : 0,
  };
}

export async function extractFeaturesFromBuffer(buffer, audioContext) {
  const channel = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  const duration = buffer.duration;

  let sumSquares = 0;
  let zeroCrossings = 0;
  let prev = channel[0] || 0;
  for (let i = 0; i < channel.length; i += 1) {
    const sample = channel[i];
    sumSquares += sample * sample;
    if ((sample >= 0 && prev < 0) || (sample < 0 && prev >= 0)) {
      zeroCrossings += 1;
    }
    prev = sample;
  }

  const rms = Math.sqrt(sumSquares / channel.length);
  const zcr = zeroCrossings / duration;

  const fftSize = 2048;
  const offline = new OfflineAudioContext(1, Math.min(channel.length, sampleRate * 4), sampleRate);
  const src = offline.createBufferSource();
  src.buffer = buffer;
  const analyserNode = offline.createAnalyser();
  analyserNode.fftSize = fftSize;
  src.connect(analyserNode);
  analyserNode.connect(offline.destination);
  src.start(0);
  await offline.startRendering();

  const freqData = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteFrequencyData(freqData);

  let weightedSum = 0;
  let totalMag = 0;
  const binWidth = sampleRate / fftSize;
  for (let i = 0; i < freqData.length; i += 1) {
    const mag = freqData[i];
    weightedSum += i * binWidth * mag;
    totalMag += mag;
  }
  const spectralCentroid = totalMag > 0 ? weightedSum / totalMag : 0;

  const rolloffTarget = totalMag * 0.85;
  let cumulative = 0;
  let spectralRolloff = 0;
  for (let i = 0; i < freqData.length; i += 1) {
    cumulative += freqData[i];
    if (cumulative >= rolloffTarget) {
      spectralRolloff = i * binWidth;
      break;
    }
  }

  const bands = bandRatios(freqData, sampleRate, fftSize);
  const f0 = estimateF0(channel, sampleRate);
  const hnr = harmonicity(channel);
  const barkEvents = detectBarkEvents(channel, sampleRate);

  return {
    vector: [
      Math.log1p(duration),
      rms,
      spectralCentroid,
      spectralRolloff,
      zcr,
      bands.low,
      bands.mid,
      bands.high,
      f0.mean,
      f0.std,
      hnr,
      barkEvents.barkCount,
      barkEvents.meanInterBarkMs,
      barkEvents.repetitionRateHz,
    ],
    summary: {
      duration,
      rms,
      spectralCentroid,
      spectralRolloff,
      zcr,
      ...bands,
      f0Mean: f0.mean,
      f0Std: f0.std,
      harmonicity: hnr,
      ...barkEvents,
    },
  };
}

export function featuresToDisplay(summary) {
  return [
    { label: "Duration", value: `${summary.duration.toFixed(2)} s` },
    { label: "Pitch (F0)", value: summary.f0Mean ? `${Math.round(summary.f0Mean)} Hz` : "Unvoiced / noisy" },
    { label: "Modulation", value: summary.f0Std ? `${Math.round(summary.f0Std)} Hz σ` : "Low" },
    { label: "Tonality", value: `${Math.round(summary.harmonicity * 100)}% harmonic` },
    { label: "Spectral centroid", value: `${Math.round(summary.spectralCentroid)} Hz` },
    { label: "Bark pulses", value: `${summary.barkCount}` },
    { label: "Repetition", value: summary.repetitionRateHz ? `${summary.repetitionRateHz.toFixed(1)} Hz` : "Single event" },
  ];
}
