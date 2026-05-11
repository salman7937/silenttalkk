window.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("video");
    const output = document.getElementById("output");
    const flipBtn = document.getElementById("flip-btn");
    const canvas = document.getElementById("canvas"); // Using the actual canvas in your HTML

    let flipped = false;
    let lastPrediction = "";
    let stableCount = 0;
    let lastAddedTime = 0;

    // Updated Urdu map with all 11 classes from your YOLO11 model
    const map = {
        "aliph": "ا",
        "bay": "ب",
        "pay": "پ",
        "ray": "ر",
        "seen": "س",
        "laam": "ل",
        "meem": "م",
        "noon": "ن",
        "kaaf": "ک",
        "hey": "ہ",
        "psl-signs": "" // Spacer or ignore
    };

    // Initialize Camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => console.error("Camera error:", err));

    // Flip functionality
    flipBtn.addEventListener("click", () => {
        flipped = !flipped;
        video.style.transform = flipped ? "scaleX(-1)" : "scaleX(1)";
        canvas.style.transform = flipped ? "scaleX(-1)" : "scaleX(1)";
    });

    // AI Detection Function
    async function predictFrame() {
        if (video.readyState !== 4) return;

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");

        // Draw current frame to canvas for visual feedback
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            const fd = new FormData();
            fd.append("image", blob, "frame.jpg");

            try {
                // Ensure this matches your Flask port (5001)
                const res = await fetch("http://127.0.0.1:5001/predict", {
                    method: "POST",
                    body: fd
                });

                const data = await res.json();
                if (!data.prediction) return;

                let current = data.prediction.toLowerCase().trim();

                // Space/Thumb support
                if (current.includes("thumb") || current.includes("up") || current === "psl-signs") {
                    if (Date.now() - lastAddedTime > 2000) {
                        output.value += " ";
                        lastAddedTime = Date.now();
                    }
                    return;
                }

                // Ignore unknown labels
                if (!map[current]) return;

                // Stability check: Model must see the same sign multiple times
                if (current === lastPrediction) {
                    stableCount++;
                } else {
                    stableCount = 1;
                    lastPrediction = current;
                }

                // Add to output if stable (sign held for ~1.5s total)
                if (stableCount >= 3 && Date.now() - lastAddedTime > 2000) {
                    output.value += map[current];
                    lastAddedTime = Date.now();
                    stableCount = 0; // Reset after adding
                }

            } catch (err) {
                console.log("Prediction Fetch Error:", err);
            }
        }, "image/jpeg");
    }

    // Increased speed: Running every 700ms for better responsiveness
    setInterval(predictFrame, 700);
});

// Clear output box
function clearText() {
    const output = document.getElementById("output");
    if (output) output.value = "";
}