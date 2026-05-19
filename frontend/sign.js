window.addEventListener("DOMContentLoaded", async () => {
    const video = document.getElementById("video");
    const output = document.getElementById("output");
    const flipBtn = document.getElementById("flip-btn");
    const canvas = document.getElementById("canvas");
    const debugPrediction = document.getElementById("debug-prediction");
    const speakBtn = document.getElementById("speak-btn");
    const stopSpeakBtn = document.getElementById("stop-speak-btn");
    const speechStatus = document.getElementById("speech-status");

    let flipped = false;
    let lastPrediction = "";
    let stableCount = 0;
    let lastAddedTime = 0;
    let lastCommittedPrediction = "";
    let noneCounter = 0;
    let predictor = null;
    let running = true;
    const hasTTS = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
    let selectedVoice = null;
    let hasUrduVoice = false;
    let speakAttemptId = 0;

    const labelMap = {
        "aliph": "ا",
        "bay": "ب",
        "pay": "پ",
        "ray": "ر",
        "seen": "س",
        "laam": "ل",
        "meem": "م",
        "noon": "ن",
        "kaaf": "ک",
        "hey": "ہ"
    };

    const speechFallbackMap = {
        "ا": "alif",
        "ب": "bay",
        "پ": "pay",
        "ر": "ray",
        "س": "seen",
        "ل": "laam",
        "م": "meem",
        "ن": "noon",
        "ک": "kaaf",
        "ہ": "hey"
    };

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            video.srcObject = stream;
            await video.play();
        } catch (err) {
            console.error("Camera error:", err);
        }
    }

    function setSpeechStatus(message) {
        if (speechStatus) speechStatus.textContent = message;
    }

    function chooseVoice() {
        if (!hasTTS) return;
        const voices = window.speechSynthesis.getVoices() || [];
        const georgeVoice =
            voices.find((voice) => voice.name && voice.name.toLowerCase() === "microsoft george") ||
            voices.find((voice) => voice.name && voice.name.toLowerCase().includes("microsoft george")) ||
            voices.find((voice) => voice.name && voice.name.toLowerCase().includes("george"));
        const urduVoice = voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith("ur"));
        const fallbackVoice =
            urduVoice ||
            voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith("hi")) ||
            voices.find((voice) => voice.lang && voice.lang.toLowerCase().startsWith("en")) ||
            voices[0] ||
            null;

        selectedVoice = georgeVoice || fallbackVoice;
        hasUrduVoice = Boolean(selectedVoice?.lang && selectedVoice.lang.toLowerCase().startsWith("ur"));
    }

    function stopSpeaking(statusMessage = "Stopped") {
        if (!hasTTS) return;
        speakAttemptId += 1;
        window.speechSynthesis.cancel();
        setSpeechStatus(statusMessage);
    }

    function normalizeSpeakText(text) {
        return text.replace(/\s+/g, " ").trim();
    }

    function buildSpeakText(rawText) {
        const normalized = normalizeSpeakText(rawText);
        if (!normalized) return "";
        const hasUrduChars = /[\u0600-\u06FF]/.test(normalized);
        if (!hasUrduVoice && hasUrduChars) {
            const fallback = normalized
                .split("")
                .map((ch) => speechFallbackMap[ch] || ch)
                .join(" ");
            return normalizeSpeakText(fallback);
        }
        return normalized;
    }

    function speakText(textToSpeak, attemptId) {
        if (!hasTTS || attemptId !== speakAttemptId || !textToSpeak) return;
        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        if (selectedVoice) {
            utterance.voice = selectedVoice;
            utterance.lang = selectedVoice.lang;
        } else {
            utterance.lang = "en-US";
        }
        utterance.rate = hasUrduVoice ? 0.95 : 0.82;
        utterance.pitch = 1;
        utterance.volume = 1;

        let started = false;
        const startedAt = Date.now();
        utterance.onstart = () => {
            started = true;
            const voiceName = selectedVoice?.name ? ` [${selectedVoice.name}]` : "";
            if (hasUrduVoice) {
                setSpeechStatus(`Speaking...${voiceName}`);
            } else {
                setSpeechStatus(`Speaking (fallback voice)...${voiceName}`);
            }
        };
        utterance.onerror = () => {
            setSpeechStatus("Speech failed");
        };
        utterance.onend = () => {
            if (attemptId !== speakAttemptId) return;
            const durationMs = Date.now() - startedAt;
            if (!started || durationMs < 120) setSpeechStatus("Speech failed");
            else setSpeechStatus("Finished");
        };

        window.speechSynthesis.speak(utterance);
    }

    function speakOutputText() {
        if (!hasTTS) return;
        chooseVoice();
        const text = (output?.value || "").trim();
        if (!text) {
            setSpeechStatus("No text to speak");
            return;
        }

        const textToSpeak = buildSpeakText(text);
        if (!textToSpeak) {
            setSpeechStatus("No text to speak");
            return;
        }

        speakAttemptId += 1;
        const attemptId = speakAttemptId;
        window.speechSynthesis.cancel();
        if (window.speechSynthesis.paused) {
            window.speechSynthesis.resume();
        }
        speakText(textToSpeak, attemptId);
    }

    if (hasTTS) {
        chooseVoice();
        if (typeof window.speechSynthesis.addEventListener === "function") {
            window.speechSynthesis.addEventListener("voiceschanged", chooseVoice);
        } else {
            window.speechSynthesis.onvoiceschanged = chooseVoice;
        }
        setSpeechStatus("Ready to speak");
    } else {
        if (speakBtn) speakBtn.disabled = true;
        if (stopSpeakBtn) stopSpeakBtn.disabled = true;
        setSpeechStatus("Text-to-speech not supported in this browser.");
    }

    speakBtn?.addEventListener("click", speakOutputText);
    stopSpeakBtn?.addEventListener("click", () => stopSpeaking("Stopped"));

    flipBtn.addEventListener("click", () => {
        flipped = !flipped;
        video.style.transform = flipped ? "scaleX(-1)" : "scaleX(1)";
        canvas.style.transform = flipped ? "scaleX(-1)" : "scaleX(1)";
    });

    function renderDebug(result) {
        if (!debugPrediction) return;
        if (!result || result.prediction === "none") {
            if (result?.reason === "no_hand") {
                debugPrediction.textContent = "No hand detected";
                return;
            }
            if (result?.reason === "pattern_miss" || result?.handDetected) {
                debugPrediction.textContent = "Hand detected but sign pattern not matched";
                return;
            }
            debugPrediction.textContent = "No hand / uncertain sign";
            return;
        }

        const top3 = result.top3 || [];
        const topText = top3.map((item) => `${item.label}:${item.confidence.toFixed(2)}`).join(" | ");
        debugPrediction.textContent = `Stable: ${stableCount} · ${result.prediction} (${result.confidence.toFixed(2)}) · ${topText}`;
    }

    function commitPrediction(prediction) {
        if (!labelMap[prediction]) return;
        const now = Date.now();
        if (now - lastAddedTime < 900) return;
        if (prediction === lastCommittedPrediction) return;

        output.value += labelMap[prediction];
        lastAddedTime = now;
        lastCommittedPrediction = prediction;
        stableCount = 0;
    }

    async function runInferenceLoop() {
        if (!predictor) return;
        const intervalMs = 150;

        while (running) {
            const result = await predictor.predict();
            renderDebug(result);

            if (result.prediction === "none") {
                noneCounter += 1;
                if (noneCounter >= 4) {
                    lastPrediction = "";
                }
            } else {
                noneCounter = 0;
            }

            if (result.prediction === "none") {
                stableCount = 0;
            } else if (result.prediction === lastPrediction) {
                stableCount += 1;
            } else {
                stableCount = 1;
                lastPrediction = result.prediction;
            }

            if (
                result.prediction !== "none" &&
                stableCount >= 5 &&
                Date.now() - lastAddedTime > 900
            ) {
                commitPrediction(result.prediction);
            }

            await new Promise((resolve) => setTimeout(resolve, intervalMs));
        }
    }

    await startCamera();

    predictor = new window.SilentTalkPredictor();
    try {
        await predictor.init(video);
        runInferenceLoop();
    } catch (err) {
        console.error("Predictor init error:", err);
        if (debugPrediction) {
            debugPrediction.textContent = "Unable to initialize MediaPipe predictor.";
        }
    }

    window.addEventListener("beforeunload", () => {
        running = false;
        predictor?.stop();
        if (hasTTS) {
            window.speechSynthesis.cancel();
        }
        const stream = video.srcObject;
        if (stream && stream.getTracks) {
            stream.getTracks().forEach((track) => track.stop());
        }
    });
});

function clearText() {
    const output = document.getElementById("output");
    if (output) output.value = "";
    if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
    }
    const speechStatus = document.getElementById("speech-status");
    if (speechStatus) speechStatus.textContent = "Cleared";
}
