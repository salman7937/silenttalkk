window.addEventListener("DOMContentLoaded", async () => {
    const video = document.getElementById("video");
    const output = document.getElementById("output");
    const flipBtn = document.getElementById("flip-btn");
    const canvas = document.getElementById("canvas");
    const debugPrediction = document.getElementById("debug-prediction");

    let flipped = false;
    let lastPrediction = "";
    let stableCount = 0;
    let lastAddedTime = 0;
    let lastCommittedPrediction = "";
    let noneCounter = 0;
    let predictor = null;
    let running = true;

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

    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            video.srcObject = stream;
            await video.play();
        } catch (err) {
            console.error("Camera error:", err);
        }
    }

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
        const stream = video.srcObject;
        if (stream && stream.getTracks) {
            stream.getTracks().forEach((track) => track.stop());
        }
    });
});

function clearText() {
    const output = document.getElementById("output");
    if (output) output.value = "";
}
