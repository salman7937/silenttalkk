window.SilentTalkFeatureExtractor = {
    distance(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = (a.z || 0) - (b.z || 0);
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    },

    dotProduct(a, b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    },

    vector(a, b) {
        return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
    },

    normalizeLandmarks(landmarks) {
        if (!landmarks || landmarks.length < 21) return [];
        const origin = landmarks[0];
        const palmSize = Math.max(this.distance(origin, landmarks[9]), 1e-6);
        return landmarks.flatMap((pt) => [
            (pt.x - origin.x) / palmSize,
            (pt.y - origin.y) / palmSize,
            ((pt.z || 0) - (origin.z || 0)) / palmSize
        ]);
    },

    jointAngle(a, b, c) {
        const ab = this.vector(b, a);
        const cb = this.vector(b, c);
        const dot = this.dotProduct(ab, cb);
        const mag = Math.max(Math.sqrt(this.dotProduct(ab, ab)) * Math.sqrt(this.dotProduct(cb, cb)), 1e-6);
        const cos = Math.min(1, Math.max(-1, dot / mag));
        return Math.acos(cos);
    },

    extractFeatureVector(landmarks) {
        if (!landmarks || landmarks.length < 21) return [];
        const normalized = this.normalizeLandmarks(landmarks);
        const wrist = landmarks[0];
        const tipIndices = [4, 8, 12, 16, 20];

        const tipDistances = tipIndices.map((idx) => this.distance(wrist, landmarks[idx]));
        const tipSpread = tipIndices.slice(1).map((idx, i) => this.distance(landmarks[tipIndices[i]], landmarks[idx]));
        const fingerAngles = [
            this.jointAngle(landmarks[2], landmarks[3], landmarks[4]),
            this.jointAngle(landmarks[5], landmarks[6], landmarks[8]),
            this.jointAngle(landmarks[9], landmarks[10], landmarks[12]),
            this.jointAngle(landmarks[13], landmarks[14], landmarks[16]),
            this.jointAngle(landmarks[17], landmarks[18], landmarks[20])
        ];

        const palmDistances = tipIndices.map((idx) => this.distance(landmarks[0], landmarks[idx]));
        return [...normalized, ...tipDistances, ...tipSpread, ...fingerAngles, ...palmDistances];
    }
};
