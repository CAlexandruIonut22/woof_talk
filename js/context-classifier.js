/**
 * Context scoring from published ethological prototypes.
 *
 * Contexts follow Molnár et al. (2008) and Pongrácz et al. (2005):
 * stranger, alone, walk, ball, play, fight
 *
 * Prototype values are normalized literature tendencies, not fitted weights.
 */

export const CONTEXTS = {
  stranger: {
    id: "stranger",
    label: "Stranger / Disturbance",
    ethology:
      "Harsh, lower-pitch, less modulated barks with longer duration and faster repetition (Yin & McCowan 2004).",
    prototype: {
      log_duration: 0.75,
      rms: 0.55,
      spectral_centroid: 900,
      spectral_rolloff: 1800,
      zero_crossing_rate: 0.08,
      low_band_ratio: 0.45,
      mid_band_ratio: 0.35,
      high_band_ratio: 0.2,
      f0_mean: 180,
      f0_std: 25,
      harmonicity: 0.35,
      bark_count: 4,
      mean_inter_bark_ms: 350,
      repetition_rate_hz: 2.8,
    },
    translations: [
      "Novel person or event detected at the territory boundary.",
      "Alert signal: something unfamiliar requires attention.",
      "Potential threat or disturbance — monitoring and signaling presence.",
    ],
    mood: "Alert / defensive",
  },
  alone: {
    id: "alone",
    label: "Isolation / Separation",
    ethology:
      "More tonal, higher-pitch, frequency-modulated vocalizations linked to separation distress (Yin & McCowan 2004; Pongrácz et al. 2005).",
    prototype: {
      log_duration: 0.55,
      rms: 0.35,
      spectral_centroid: 1400,
      spectral_rolloff: 2600,
      zero_crossing_rate: 0.06,
      low_band_ratio: 0.25,
      mid_band_ratio: 0.45,
      high_band_ratio: 0.3,
      f0_mean: 320,
      f0_std: 70,
      harmonicity: 0.62,
      bark_count: 3,
      mean_inter_bark_ms: 700,
      repetition_rate_hz: 1.4,
    },
    translations: [
      "Social contact is missing — calling for reunion with the group.",
      "Separation-related vocalization; likely seeking owner or companion return.",
      "Isolation distress signal with tonal, attention-seeking quality.",
    ],
    mood: "Distress / contact-seeking",
  },
  walk: {
    id: "walk",
    label: "Anticipation / Walk",
    ethology:
      "Positive anticipation vocalizations before a rewarding activity (Molnár et al. 2008 'Walk' context).",
    prototype: {
      log_duration: 0.35,
      rms: 0.42,
      spectral_centroid: 1250,
      spectral_rolloff: 2400,
      zero_crossing_rate: 0.07,
      low_band_ratio: 0.28,
      mid_band_ratio: 0.42,
      high_band_ratio: 0.3,
      f0_mean: 280,
      f0_std: 55,
      harmonicity: 0.55,
      bark_count: 2,
      mean_inter_bark_ms: 500,
      repetition_rate_hz: 2.0,
    },
    translations: [
      "Positive anticipation — a familiar rewarding activity is about to begin.",
      "Excited readiness signal, often before walks or outdoor activity.",
      "Motivated solicitation tied to an expected routine reward.",
    ],
    mood: "Anticipatory / excited",
  },
  ball: {
    id: "ball",
    label: "Solicitation / Resource Request",
    ethology:
      "Repeated barks directed at a withheld toy or resource (Molnár 'Ball' context; Pongrácz 'asking for ball').",
    prototype: {
      log_duration: 0.45,
      rms: 0.48,
      spectral_centroid: 1150,
      spectral_rolloff: 2200,
      zero_crossing_rate: 0.075,
      low_band_ratio: 0.3,
      mid_band_ratio: 0.4,
      high_band_ratio: 0.3,
      f0_mean: 250,
      f0_std: 45,
      harmonicity: 0.5,
      bark_count: 5,
      mean_inter_bark_ms: 420,
      repetition_rate_hz: 2.4,
    },
    translations: [
      "Resource-directed solicitation — requesting access to toy, food, or object.",
      "Persistent request signal aimed at a human gatekeeper.",
      "Goal-oriented barking toward a desired item or outcome.",
    ],
    mood: "Soliciting / persistent",
  },
  play: {
    id: "play",
    label: "Play",
    ethology:
      "Harmonically richer, modulated vocalizations during social play (Yin & McCowan 2004; Molnár 'Play').",
    prototype: {
      log_duration: 0.4,
      rms: 0.4,
      spectral_centroid: 1500,
      spectral_rolloff: 2800,
      zero_crossing_rate: 0.065,
      low_band_ratio: 0.22,
      mid_band_ratio: 0.43,
      high_band_ratio: 0.35,
      f0_mean: 340,
      f0_std: 90,
      harmonicity: 0.68,
      bark_count: 3,
      mean_inter_bark_ms: 550,
      repetition_rate_hz: 1.8,
    },
    translations: [
      "Social play vocalization — affiliative, modulated, non-aggressive.",
      "Play-invitation or play-arousal signal during interaction.",
      "Positive social engagement; likely part of a play bout.",
    ],
    mood: "Playful / affiliative",
  },
  fight: {
    id: "fight",
    label: "Agonistic / Threat",
    ethology:
      "Low-pitched, harsh, low harmonicity barks associated with aggression contexts (Pongrácz et al. 2005).",
    prototype: {
      log_duration: 0.3,
      rms: 0.62,
      spectral_centroid: 750,
      spectral_rolloff: 1500,
      zero_crossing_rate: 0.09,
      low_band_ratio: 0.5,
      mid_band_ratio: 0.32,
      high_band_ratio: 0.18,
      f0_mean: 150,
      f0_std: 20,
      harmonicity: 0.28,
      bark_count: 4,
      mean_inter_bark_ms: 280,
      repetition_rate_hz: 3.5,
    },
    translations: [
      "Agonistic or threat-related vocalization — escalation warning.",
      "Defensive or offensive signaling during conflict context.",
      "High-intensity bark with harsh spectral profile; treat with caution.",
    ],
    mood: "Agonistic / threat",
  },
};

const FEATURE_KEYS = [
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

function featureObjectFromVector(vector) {
  return FEATURE_KEYS.reduce((acc, key, index) => {
    acc[key] = vector[index];
    return acc;
  }, {});
}

function normalizeFeatures(features) {
  const scales = {
    log_duration: 1,
    rms: 1,
    spectral_centroid: 0.001,
    spectral_rolloff: 0.001,
    zero_crossing_rate: 10,
    low_band_ratio: 1,
    mid_band_ratio: 1,
    high_band_ratio: 1,
    f0_mean: 0.005,
    f0_std: 0.01,
    harmonicity: 1,
    bark_count: 0.25,
    mean_inter_bark_ms: 0.001,
    repetition_rate_hz: 0.2,
  };

  const normalized = {};
  for (const key of FEATURE_KEYS) {
    normalized[key] = features[key] * (scales[key] || 1);
  }
  return normalized;
}

function distance(a, b) {
  let sum = 0;
  for (const key of FEATURE_KEYS) {
    const delta = a[key] - b[key];
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

export function classifyContext(featureVector) {
  const features = normalizeFeatures(featureObjectFromVector(featureVector));
  const scores = Object.values(CONTEXTS).map((context) => {
    const proto = normalizeFeatures(context.prototype);
    const dist = distance(features, proto);
    const confidence = 1 / (1 + dist);
    return { context, dist, confidence };
  });

  scores.sort((a, b) => b.confidence - a.confidence);
  const best = scores[0];
  const second = scores[1];
  const margin = best.confidence - second.confidence;
  const normalizedConfidence = Math.min(0.92, 0.45 + best.confidence * 0.35 + margin * 0.4);

  const line =
    best.context.translations[Math.floor(Math.random() * best.context.translations.length)];

  return {
    id: best.context.id,
    label: best.context.label,
    ethology: best.context.ethology,
    mood: best.context.mood,
    line,
    confidence: normalizedConfidence,
    ranking: scores.slice(0, 3).map((entry) => ({
      label: entry.context.label,
      score: entry.confidence,
    })),
  };
}

export function composeTranslation(contextResult, emotionResult) {
  const arousal = emotionResult?.arousal?.label || "Medium";
  const valence = emotionResult?.valence?.label || "Neutral";

  const arousalNote =
    arousal === "High"
      ? "High arousal suggests elevated motor activation."
      : arousal === "Low"
        ? "Low arousal suggests subdued energy or recovery state."
        : "Moderate arousal suggests engaged but controlled signaling.";

  const valenceNote =
    valence === "Negative"
      ? "Negative valence aligns with unpleasant or aversive motivation."
      : valence === "Positive"
        ? "Positive valence aligns with appetitive or affiliative motivation."
        : "Neutral valence may indicate informational signaling without strong affect.";

  return {
    primary: contextResult.line,
    secondary: `${contextResult.ethology} ${arousalNote} ${valenceNote} (EmotionalCanines framework; Dang et al. 2025).`,
  };
}
