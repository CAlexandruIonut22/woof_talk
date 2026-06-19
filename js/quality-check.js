export function assessAudioQuality(summary) {
  const issues = [];
  let score = 100;

  if (!summary) {
    return {
      score: 0,
      label: "No audio analysis",
      pass: false,
      issues: ["No acoustic summary was generated."]
    };
  }

  if (summary.duration < 0.6) {
    score -= 35;
    issues.push("Recording is very short. Try capturing at least 1–3 seconds.");
  }

  if (summary.duration > 20) {
    score -= 10;
    issues.push("Recording is long. Shorter clips with one clear vocalization work better.");
  }

  if (summary.rms < 0.012) {
    score -= 30;
    issues.push("Audio is quiet. Move the microphone closer to the dog.");
  }

  if (summary.barkCount === 0 && !summary.f0Mean) {
    score -= 30;
    issues.push("No clear bark, whine, or growl pattern was detected.");
  }

  if (summary.high > 0.55 && summary.rms < 0.04) {
    score -= 15;
    issues.push("Possible background hiss or environmental noise detected.");
  }

  if (summary.harmonicity < 0.12 && summary.rms < 0.03) {
    score -= 15;
    issues.push("Signal is noisy or unclear.");
  }

  const finalScore = Math.max(0, Math.min(100, Math.round(score)));

  let label = "Good recording";
  if (finalScore < 45) label = "Poor recording";
  else if (finalScore < 70) label = "Usable but uncertain";

  return {
    score: finalScore,
    label,
    pass: finalScore >= 45,
    issues
  };
}