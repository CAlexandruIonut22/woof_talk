import {
  extractFeaturesFromBuffer,
  featuresToDisplay,
} from "./js/acoustic-features.js";
import { classifyContext, composeTranslation } from "./js/context-classifier.js";
import {
  fallbackEmotionFromFeatures,
  loadEmotionModel,
  predictEmotion,
} from "./js/emotion-model.js";
import { assessAudioQuality } from "./js/quality-check.js";
import {
  getDogProfile,
  getSceneContext,
  hydrateDogProfile,
  saveDogProfile
} from "./js/profile-context.js";
import { requestAiInterpretation } from "./js/ai-client.js";

const recordBtn = document.getElementById("recordBtn");
const recordLabel = document.getElementById("recordLabel");
const recordIcon = document.getElementById("recordIcon");
const fileInput = document.getElementById("fileInput");
const clearBtn = document.getElementById("clearBtn");
const micStatus = document.getElementById("micStatus");
const waveformCanvas = document.getElementById("waveform");
const waveformHint = document.getElementById("waveformHint");
const dogAvatar = document.getElementById("dogAvatar");
const thoughtBubble = document.getElementById("thoughtBubble");
const resultPanel = document.getElementById("resultPanel");
const detectedType = document.getElementById("detectedType");
const confidence = document.getElementById("confidence");
const mood = document.getElementById("mood");
const arousalValue = document.getElementById("arousalValue");
const valenceValue = document.getElementById("valenceValue");
const translationText = document.getElementById("translationText");
const translationSubtext = document.getElementById("translationSubtext");
const featureGrid = document.getElementById("featureGrid");
const rankingList = document.getElementById("rankingList");
const modelMeta = document.getElementById("modelMeta");
const translateAgainBtn = document.getElementById("translateAgainBtn");
const shareBtn = document.getElementById("shareBtn");
const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const toast = document.getElementById("toast");
const profileSaveBtn = document.getElementById("profileSaveBtn");
const qualityStatus = document.getElementById("qualityStatus");
const aiCard = document.getElementById("aiCard");
const aiConfidence = document.getElementById("aiConfidence");
const aiMeaning = document.getElementById("aiMeaning");
const aiPhrase = document.getElementById("aiPhrase");
const aiWhy = document.getElementById("aiWhy");
const aiPossibilities = document.getElementById("aiPossibilities");
const aiAction = document.getElementById("aiAction");
const aiWarnings = document.getElementById("aiWarnings");
const aiDisclaimer = document.getElementById("aiDisclaimer");

const canvasCtx = waveformCanvas.getContext("2d");
const HISTORY_KEY = "woof-talk-history-v2";

let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let currentAudioBuffer = null;
let currentBlobUrl = null;
let isRecording = false;
let animationFrame = null;
let analyser = null;
let sourceNode = null;
let lastTranslation = null;
let emotionModel = null;

const DATASETS = [
  {
    name: "EmotionalCanines",
    size: "1,400 bark sequences",
    labels: "Arousal (Low/Medium/High), Valence (Negative/Neutral/Positive)",
    url: "https://huggingface.co/datasets/ArlingtonCL2/BarkopediaDogEmotionClassification_Data",
    use: "Emotion model in this app",
  },
  {
    name: "DogSpeak",
    size: "77,202 bark sequences · 156 dogs · 5 breeds",
    labels: "Breed, sex, individual dog ID",
    url: "https://huggingface.co/datasets/ArlingtonCL2/DogSpeak_Dataset",
    use: "Large-scale bioacoustics benchmarks",
  },
  {
    name: "Barkopedia Breed",
    size: "29,347 labeled clips",
    labels: "Dog breed classification",
    url: "https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_DOG_BREED_CLASSIFICATION_DATASET",
    use: "Breed recognition training",
  },
  {
    name: "Molnár et al. (2008)",
    size: "6,000+ context-labeled barks",
    labels: "Stranger, Alone, Walk, Ball, Play, Fight",
    url: "https://infoscience.epfl.ch/entities/publication/c98910c1-4f5b-4386-ad3a-9c13108e141b",
    use: "Context prototype design in this app",
  },
];

function renderDatasetCatalog() {
  const catalog = document.getElementById("datasetCatalog");
  if (!catalog) return;
  catalog.innerHTML = DATASETS.map(
    (dataset) => `
      <article class="dataset-card">
        <h3>${dataset.name}</h3>
        <p><strong>Scale:</strong> ${dataset.size}</p>
        <p><strong>Labels:</strong> ${dataset.labels}</p>
        <p><strong>Used for:</strong> ${dataset.use}</p>
        <a href="${dataset.url}" target="_blank" rel="noopener noreferrer">View dataset</a>
      </article>
    `,
  ).join("");
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.add("hidden"), 2400);
}

function setDogState(state, bubbleText = "...") {
  dogAvatar.classList.remove("listening", "thinking");
  if (state === "listening") dogAvatar.classList.add("listening");
  if (state === "thinking") dogAvatar.classList.add("thinking");
  thoughtBubble.textContent = bubbleText;
  thoughtBubble.classList.toggle("visible", bubbleText !== "...");
}

function drawFlatWaveform(message) {
  canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  canvasCtx.fillStyle = "#6b5b4f";
  canvasCtx.font = "14px Inter, sans-serif";
  canvasCtx.textAlign = "center";
  canvasCtx.fillText(message, waveformCanvas.width / 2, waveformCanvas.height / 2);
}

function drawWaveform(dataArray) {
  const { width, height } = waveformCanvas;
  canvasCtx.clearRect(0, 0, width, height);
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = "#ff7a18";
  canvasCtx.beginPath();

  const sliceWidth = width / dataArray.length;
  let x = 0;
  for (let i = 0; i < dataArray.length; i += 1) {
    const v = dataArray[i] / 128.0;
    const y = (v * height) / 2;
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
    x += sliceWidth;
  }
  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();
}

function animateLiveWaveform() {
  if (!analyser) return;
  const dataArray = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(dataArray);
  drawWaveform(dataArray);
  animationFrame = requestAnimationFrame(animateLiveWaveform);
}

function stopLiveWaveform() {
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = null;
}

async function ensureAudioContext() {
  if (!audioContext) audioContext = new AudioContext();
  if (audioContext.state === "suspended") await audioContext.resume();
  return audioContext;
}

function revokeCurrentBlobUrl() {
  if (currentBlobUrl) {
    URL.revokeObjectURL(currentBlobUrl);
    currentBlobUrl = null;
  }
}

async function drawBufferWaveform(buffer) {
  const channel = buffer.getChannelData(0);
  const step = Math.ceil(channel.length / waveformCanvas.width);
  const dataArray = new Uint8Array(waveformCanvas.width);

  for (let i = 0; i < waveformCanvas.width; i += 1) {
    const start = i * step;
    let min = 1;
    let max = -1;
    for (let j = 0; j < step && start + j < channel.length; j += 1) {
      const sample = channel[start + j];
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    dataArray[i] = 128 + ((min + max) / 2) * 127;
  }
  drawWaveform(dataArray);
}

function renderQualityStatus(quality) {
  if (!qualityStatus) return;

  qualityStatus.classList.remove("hidden", "good", "warning", "bad");

  if (quality.score >= 70) {
    qualityStatus.classList.add("good");
  } else if (quality.score >= 45) {
    qualityStatus.classList.add("warning");
  } else {
    qualityStatus.classList.add("bad");
  }

  const issueHtml = quality.issues.length
    ? `<ul>${quality.issues.map((issue) => `<li>${issue}</li>`).join("")}</ul>`
    : "<p>Audio quality looks good for analysis.</p>";

  qualityStatus.innerHTML = `
    <strong>${quality.label} · ${quality.score}/100</strong>
    ${issueHtml}
  `;
}

function renderAiLoading() {
  if (!aiCard) return;

  aiCard.classList.remove("hidden");
  aiConfidence.textContent = "Confidence —";
  aiMeaning.textContent = "Generating AI interpretation...";
  aiPhrase.textContent = "";
  aiWhy.innerHTML = "";
  aiPossibilities.innerHTML = "";
  aiAction.textContent = "";
  aiWarnings.classList.add("hidden");
  aiWarnings.innerHTML = "";
  aiDisclaimer.textContent = "";
}

function renderAiError(message) {
  if (!aiCard) return;

  aiCard.classList.remove("hidden");
  aiConfidence.textContent = "Confidence low";
  aiMeaning.textContent = "AI interpretation unavailable";
  aiPhrase.textContent = "The app still completed the acoustic analysis, but the LLM interpretation could not be generated.";
  aiWhy.innerHTML = `<li>${message}</li>`;
  aiPossibilities.innerHTML = "";
  aiAction.textContent = "Try again, or check that the Vercel environment variable OPENAI_API_KEY is set.";
  aiWarnings.classList.add("hidden");
  aiWarnings.innerHTML = "";
  aiDisclaimer.textContent = "This tool is not a veterinary diagnosis or literal dog-language translator.";
}

function renderAiResult(ai) {
  if (!aiCard || !ai) return;

  aiCard.classList.remove("hidden");

  aiConfidence.textContent = `Confidence ${ai.confidenceLevel} · ${Math.round(
    ai.confidenceScore * 100
  )}%`;

  aiMeaning.textContent = ai.likelyMeaning;
  aiPhrase.textContent = `“${ai.humanPhrase}”`;

  aiWhy.innerHTML = (ai.why || [])
    .map((item) => `<li>${item}</li>`)
    .join("");

  aiPossibilities.innerHTML = (ai.possibleMeanings || [])
    .map(
      (item) => `
        <li>
          <strong>${item.label}</strong>
          <span>${Math.round(item.probability * 100)}%</span>
          <small>${item.reason}</small>
        </li>
      `
    )
    .join("");

  aiAction.textContent = ai.recommendedAction || "";

  if (ai.warnings?.length) {
    aiWarnings.classList.remove("hidden");
    aiWarnings.innerHTML = `
      <strong>Important</strong>
      <ul>${ai.warnings.map((warning) => `<li>${warning}</li>`).join("")}</ul>
    `;
  } else {
    aiWarnings.classList.add("hidden");
    aiWarnings.innerHTML = "";
  }

  aiDisclaimer.textContent =
    ai.disclaimer ||
    "This is an AI-assisted interpretation, not a literal translation or veterinary diagnosis.";
}

function saveFeedback(rating) {
  if (!lastTranslation) return;

  const key = "woof-talk-feedback-v1";

  let entries = [];

  try {
    entries = JSON.parse(localStorage.getItem(key) || "[]");
  } catch {
    entries = [];
  }

  entries.unshift({
    rating,
    createdAt: new Date().toISOString(),
    context: lastTranslation.contextResult?.label,
    arousal: lastTranslation.emotionResult?.arousal?.label,
    valence: lastTranslation.emotionResult?.valence?.label,
    aiMeaning: lastTranslation.ai?.likelyMeaning || "",
    dogProfile: lastTranslation.dogProfile || {},
    sceneContext: lastTranslation.sceneContext || {},
    quality: lastTranslation.quality || null
  });

  localStorage.setItem(key, JSON.stringify(entries.slice(0, 50)));
  showToast("Feedback saved. This can be used later for personalization.");
}

async function translateCurrentAudio() {
  if (!currentAudioBuffer) return;

  setDogState("thinking", "...");
  resultPanel.classList.add("hidden");
  waveformHint.textContent = "Extracting acoustic features...";

  await new Promise((resolve) => setTimeout(resolve, 500));

  const ctx = await ensureAudioContext();
  const { vector, summary } = await extractFeaturesFromBuffer(currentAudioBuffer, ctx);

  const quality = assessAudioQuality(summary);
  const dogProfile = getDogProfile();
  const sceneContext = getSceneContext();

  renderQualityStatus(quality);

  const contextResult = classifyContext(vector);

  let emotionResult = null;

  try {
    emotionResult = emotionModel
      ? predictEmotion(emotionModel, vector)
      : fallbackEmotionFromFeatures(summary);
  } catch {
    emotionResult = fallbackEmotionFromFeatures(summary);
  }

  const composed = composeTranslation(contextResult, emotionResult);

  lastTranslation = {
    contextResult,
    emotionResult,
    composed,
    summary,
    quality,
    dogProfile,
    sceneContext,
    ai: null
  };

  detectedType.textContent = contextResult.label;
  confidence.textContent = `${Math.round(contextResult.confidence * 100)}%`;
  mood.textContent = contextResult.mood;

  arousalValue.textContent = `${emotionResult.arousal.label} (${Math.round(
    emotionResult.arousal.confidence * 100
  )}%)`;

  valenceValue.textContent = `${emotionResult.valence.label} (${Math.round(
    emotionResult.valence.confidence * 100
  )}%)`;

  translationText.textContent = composed.primary;
  translationSubtext.textContent = composed.secondary;

  featureGrid.innerHTML = featuresToDisplay(summary)
    .map(
      (item) => `
        <div class="feature-chip">
          <span class="feature-chip-label">${item.label}</span>
          <span class="feature-chip-value">${item.value}</span>
        </div>
      `
    )
    .join("");

  rankingList.innerHTML = contextResult.ranking
    .map(
      (entry) => `
        <li>
          <span>${entry.label}</span>
          <strong>${Math.round(entry.score * 100)}%</strong>
        </li>
      `
    )
    .join("");

  if (emotionResult.meta?.crossValidation?.note) {
    modelMeta.textContent = emotionResult.meta.crossValidation.note;
  } else if (emotionResult.meta?.crossValidation?.arousalAccuracy) {
    modelMeta.textContent = `Emotion model trained on ${
      emotionResult.meta.samplesTrained
    } clips from EmotionalCanines. CV accuracy — arousal ${Math.round(
      emotionResult.meta.crossValidation.arousalAccuracy * 100
    )}%, valence ${Math.round(
      emotionResult.meta.crossValidation.valenceAccuracy * 100
    )}%.`;
  } else if (emotionResult.meta?.fallback) {
    modelMeta.textContent =
      "Emotion model unavailable — using literature-based heuristic fallback. Run ml/train_emotion_classifier.py to load the trained model.";
  } else {
    modelMeta.textContent = emotionModel
      ? `Emotion model loaded (${emotionModel.dataset}).`
      : "Loading emotion model...";
  }

  resultPanel.classList.remove("hidden");
  setDogState("idle", "!");
  waveformHint.textContent = "Acoustic analysis complete";
  clearBtn.disabled = false;

  renderAiLoading();

  const aiPayload = {
    acousticSummary: summary,
    audioQuality: quality,
    contextPrediction: {
      id: contextResult.id,
      label: contextResult.label,
      confidence: contextResult.confidence,
      mood: contextResult.mood,
      ethology: contextResult.ethology,
      ranking: contextResult.ranking
    },
    emotionPrediction: {
      arousal: emotionResult.arousal,
      valence: emotionResult.valence
    },
    dogProfile,
    sceneContext,
    safetyFlags: {
      possiblePainOrUnusualBehavior:
        sceneContext.trigger === "pain_or_unusual" ||
        sceneContext.bodyLanguage.includes("hiding"),
      growlingReported: sceneContext.bodyLanguage.includes("growling"),
      lowAudioQuality: !quality.pass
    }
  };

  try {
    if (!quality.pass) {
      aiPayload.systemNote =
        "The audio quality is poor. The interpretation should be cautious and low-confidence.";
    }

    const ai = await requestAiInterpretation(aiPayload);
    lastTranslation.ai = ai;
    renderAiResult(ai);
    waveformHint.textContent = "AI interpretation complete";
  } catch (error) {
    renderAiError(error.message);
    waveformHint.textContent = "Acoustic analysis complete · AI unavailable";
  }

  addHistoryEntry(lastTranslation);
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  mediaRecorder.stop();
  isRecording = false;
  recordBtn.classList.remove("recording");
  recordLabel.textContent = "Hold to Record";
  recordIcon.textContent = "🎙️";
  micStatus.textContent = "Processing vocalization...";
}

function clearAudio() {
  stopRecording();
  revokeCurrentBlobUrl();
  currentAudioBuffer = null;
  lastTranslation = null;
  resultPanel.classList.add("hidden");
  clearBtn.disabled = true;
  waveformHint.textContent = "Record or upload a bark, whine, or growl";
  setDogState("idle", "...");
  drawFlatWaveform("Waiting for audio");
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(entries) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, 8)));
}

function renderHistory() {
  const entries = loadHistory();
  historyList.innerHTML = "";
  if (entries.length === 0) {
    historyList.innerHTML =
      '<li class="history-empty">No analyses yet. Record a vocalization to begin.</li>';
    return;
  }

  entries.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "history-item";
    item.innerHTML = `
      <span class="history-type">${entry.type}</span>
      <span class="history-quote">${entry.line}</span>
      <span class="history-meta">${entry.time} · ${entry.arousal}/${entry.valence}</span>
    `;
    historyList.appendChild(item);
  });
}

function addHistoryEntry(result) {
  const entries = loadHistory();
  entries.unshift({
    type: result.contextResult.label,
    line: result.composed.primary,
    arousal: result.emotionResult.arousal.label,
    valence: result.emotionResult.valence.label,
    time: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
  });
  saveHistory(entries);
  renderHistory();
}

recordBtn.addEventListener("mousedown", startRecording);
recordBtn.addEventListener("mouseup", stopRecording);
recordBtn.addEventListener("mouseleave", stopRecording);
recordBtn.addEventListener("touchstart", (event) => {
  event.preventDefault();
  startRecording();
});
recordBtn.addEventListener("touchend", (event) => {
  event.preventDefault();
  stopRecording();
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  micStatus.textContent = `Loaded ${file.name}`;
  await handleAudioBlob(file);
  fileInput.value = "";
});

clearBtn.addEventListener("click", clearAudio);
translateAgainBtn.addEventListener("click", () => {
  if (currentAudioBuffer) translateCurrentAudio();
});

shareBtn.addEventListener("click", async () => {
  if (!lastTranslation) return;
  const text = [
    `Interpretation: ${
  lastTranslation.ai?.humanPhrase ||
  lastTranslation.ai?.likelyMeaning ||
  lastTranslation.composed.primary
}`,
  ].join("\n");
  try {
    await navigator.clipboard.writeText(text);
    showToast("Analysis copied.");
  } catch {
    showToast("Could not copy to clipboard.");
  }
});

clearHistoryBtn.addEventListener("click", () => {
  saveHistory([]);
  renderHistory();
  showToast("History cleared.");
});

profileSaveBtn?.addEventListener("click", () => {
  saveDogProfile();
  showToast("Dog profile saved.");
});

document.querySelectorAll("[data-feedback]").forEach((button) => {
  button.addEventListener("click", () => {
    saveFeedback(button.dataset.feedback);
  });
});
async function bootstrap() {
  hydrateDogProfile();
  renderDatasetCatalog();
  renderHistory();
  drawFlatWaveform("Waiting for audio");
  try {
    emotionModel = await loadEmotionModel();
    micStatus.textContent = "Microphone ready · emotion model loaded";
  } catch {
    micStatus.textContent = "Microphone ready · run ml/train_emotion_classifier.py for ML emotion model";
  }
}

bootstrap();
