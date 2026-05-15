window.addEventListener("DOMContentLoaded", async () => {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const labelSelect = document.getElementById("label-select");
    const captureBtn = document.getElementById("capture-btn");
    const downloadBtn = document.getElementById("download-btn");
    const clearBtn = document.getElementById("clear-btn");
    const status = document.getElementById("status");
    const poseStatus = document.getElementById("pose-status");
    const STORAGE_KEY = "silenttalk_landmark_samples_v1";

    let currentLandmarks = null;
    const samples = [];
    let handVisible = false;

    function saveSamples() {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
    }

    function refreshStatus() {
        status.textContent = `Samples collected: ${samples.length}`;
        downloadBtn.disabled = samples.length === 0;
    }

    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            if (Array.isArray(parsed)) {
                samples.push(...parsed);
            }
        }
    } catch (error) {
        console.warn("Failed to restore local samples:", error);
    }

    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    });

    hands.onResults((results) => {
        const ctx = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            handVisible = true;
            currentLandmarks = results.multiHandLandmarks[0].map((landmark) => ({ x: landmark.x, y: landmark.y, z: landmark.z }));
            poseStatus.textContent = "Hand detected. Ready to capture.";
            drawConnectors(ctx, results.multiHandLandmarks[0], HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
            drawLandmarks(ctx, results.multiHandLandmarks[0], { color: '#FF0000', lineWidth: 1 });
        } else {
            handVisible = false;
            currentLandmarks = null;
            poseStatus.textContent = "No hand detected. Please move your hand into view.";
        }
    });

    const camera = new Camera(video, {
        onFrame: async () => await hands.send({ image: video }),
        width: 640,
        height: 480
    });

    await camera.start();

    captureBtn.addEventListener("click", () => {
        if (!handVisible || !currentLandmarks) {
            alert("Please show a clear hand pose before capturing.");
            return;
        }

        const label = labelSelect.value;
        samples.push({
            label,
            landmarks: currentLandmarks,
            timestamp: new Date().toISOString()
        });

        saveSamples();
        refreshStatus();
    });

    downloadBtn.addEventListener("click", () => {
        if (!samples.length) return;

        const blob = new Blob([JSON.stringify(samples, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `silenttalk_landmark_dataset_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    });

    clearBtn.addEventListener("click", () => {
        if (!samples.length) return;
        if (!window.confirm("Delete all locally stored samples?")) return;

        samples.splice(0, samples.length);
        localStorage.removeItem(STORAGE_KEY);
        refreshStatus();
    });

    refreshStatus();
});
