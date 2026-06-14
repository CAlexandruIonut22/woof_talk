/**
 * Browser inference for logistic-regression models exported from ml/train_emotion_classifier.py
 */

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((value) => value / sum);
}

function scaleFeatures(vector, mean, scale) {
  return vector.map((value, index) => (value - mean[index]) / (scale[index] || 1));
}

function predictBlock(vector, block) {
  const scaled = scaleFeatures(vector, block.mean, block.scale);
  const logits = block.coefficients.map((row, classIndex) => {
    const dot = row.reduce((sum, weight, featureIndex) => sum + weight * scaled[featureIndex], 0);
    return dot + block.intercept[classIndex];
  });
  const probabilities = softmax(logits);
  const bestIndex = probabilities.indexOf(Math.max(...probabilities));
  return {
    label: block.classes[bestIndex],
    confidence: probabilities[bestIndex],
    distribution: block.classes.map((label, index) => ({
      label,
      probability: probabilities[index],
    })),
  };
}

export async function loadEmotionModel() {
  const response = await fetch("./models/emotion-classifier.json");
  if (!response.ok) {
    throw new Error("Emotion model not found. Run ml/train_emotion_classifier.py first.");
  }
  return response.json();
}

export function predictEmotion(model, featureVector) {
  if (!model || featureVector.length !== model.featureNames.length) {
    return null;
  }

  return {
    arousal: predictBlock(featureVector, model.arousal),
    valence: predictBlock(featureVector, model.valence),
    meta: {
      dataset: model.dataset,
      citation: model.citation,
      samplesTrained: model.samplesTrained,
      crossValidation: model.crossValidation,
    },
  };
}

export function fallbackEmotionFromFeatures(summary) {
  const arousalScore = summary.rms * 2 + summary.repetitionRateHz * 0.15 + summary.barkCount * 0.08;
  const valenceScore = summary.harmonicity + summary.f0Std * 0.002 - summary.low * 0.25;

  const arousal =
    arousalScore > 0.75 ? "High" : arousalScore > 0.4 ? "Medium" : "Low";
  const valence =
    valenceScore > 0.55 ? "Positive" : valenceScore < 0.35 ? "Negative" : "Neutral";

  return {
    arousal: { label: arousal, confidence: 0.55, distribution: [] },
    valence: { label: valence, confidence: 0.55, distribution: [] },
    meta: { fallback: true },
  };
}
