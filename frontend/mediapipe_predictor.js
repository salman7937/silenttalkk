window.SilentTalkPredictor = class {
    constructor(config = {}) {
        this.config = Object.assign({
            maxHands: 1,
            minDetectionConfidence: 0.45,
            minTrackingConfidence: 0.45,
            inferenceEndpoint: "/predict_landmarks"
        }, config);

        this.latestResult = null;
        this.videoElement = null;
        this.rafId = null;
        this.isRunning = false;
        this.hands = null;
        this.requestInFlight = false;
    }

    async init(videoElement) {
        if (!window.Hands) {
            throw new Error("MediaPipe Hands is not loaded.");
        }

        this.hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        this.hands.setOptions({
            maxNumHands: this.config.maxHands,
            modelComplexity: 1,
            minDetectionConfidence: this.config.minDetectionConfidence,
            minTrackingConfidence: this.config.minTrackingConfidence
        });

        this.hands.onResults((results) => {
            this.latestResult = results;
        });

        this.videoElement = videoElement;
        this.isRunning = true;
        this.startFrameLoop();
    }

    stop() {
        this.isRunning = false;
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
    }

    startFrameLoop() {
        const tick = async () => {
            if (!this.isRunning || !this.hands || !this.videoElement) return;

            if (this.videoElement.readyState >= 2) {
                try {
                    await this.hands.send({ image: this.videoElement });
                } catch (err) {
                    console.error("MediaPipe frame processing error:", err);
                }
            }

            this.rafId = requestAnimationFrame(() => {
                tick();
            });
        };

        tick();
    }

    getLatestLandmarks() {
        if (!this.latestResult || !this.latestResult.multiHandLandmarks || !this.latestResult.multiHandLandmarks.length) {
            return null;
        }
        return this.latestResult.multiHandLandmarks[0];
    }

    async predict() {
        const landmarks = this.getLatestLandmarks();
        if (!landmarks) {
            return { prediction: "none", confidence: 0, top3: [], handDetected: false, reason: "no_hand" };
        }

        if (this.requestInFlight) {
            return { prediction: "none", confidence: 0, top3: [], handDetected: true, reason: "inference_busy" };
        }

        this.requestInFlight = true;
        try {
            const response = await fetch(this.config.inferenceEndpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ landmarks })
            });

            if (!response.ok) {
                return { prediction: "none", confidence: 0, top3: [], handDetected: true, reason: "server_error" };
            }

            const data = await response.json();
            return {
                prediction: data.prediction || "none",
                confidence: Number(data.confidence || 0),
                top3: Array.isArray(data.top3) ? data.top3 : [],
                handDetected: true
            };
        } catch (error) {
            console.error("Prediction request failed:", error);
            return { prediction: "none", confidence: 0, top3: [], handDetected: true, reason: "network_error" };
        } finally {
            this.requestInFlight = false;
        }
    }
};
