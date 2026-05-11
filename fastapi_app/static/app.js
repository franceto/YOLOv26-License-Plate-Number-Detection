const video = document.getElementById("video");
const videoInput = document.getElementById("videoInput");
const overlay = document.getElementById("overlay");
const sendCanvas = document.getElementById("sendCanvas");

const ctx = overlay.getContext("2d");
const sendCtx = sendCanvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const ocrBtn = document.getElementById("ocrBtn");

const conf = document.getElementById("conf");
const confVal = document.getElementById("confVal");
const imgsz = document.getElementById("imgsz");

const statusEl = document.getElementById("status");
const latencyEl = document.getElementById("latency");
const topCountEl = document.getElementById("topCount");
const delayEl = document.getElementById("delay");
const ocrResults = document.getElementById("ocrResults");

let running = false;
let busy = false;
let sessionId = crypto.randomUUID();
let lastSent = 0;
let autoDelay = 160;
let lastBoxes = [];

conf.addEventListener("input", () => {
    confVal.textContent = Number(conf.value).toFixed(2);
});

function setStatus(text) {
    statusEl.textContent = text;
}

function clearOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function resizeOverlay() {
    if (!video.videoWidth || !video.videoHeight) return;

    const rect = video.getBoundingClientRect();

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;

    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;

    clearOverlay();

    if (lastBoxes.length > 0) {
        drawBoxes(lastBoxes, video.videoWidth, video.videoHeight);
    }
}

function drawBoxes(boxes, frameW, frameH) {
    clearOverlay();

    if (!frameW || !frameH) return;

    const sx = overlay.width / frameW;
    const sy = overlay.height / frameH;

    ctx.lineWidth = Math.max(2, overlay.width / 500);
    ctx.font = `${Math.max(16, overlay.width / 60)}px Arial`;
    ctx.textBaseline = "top";

    boxes.forEach(b => {
        const x = b.x1 * sx;
        const y = b.y1 * sy;
        const w = (b.x2 - b.x1) * sx;
        const h = (b.y2 - b.y1) * sy;

        if (w < 8 || h < 8) return;

        ctx.strokeStyle = "#22c55e";
        ctx.fillStyle = "#22c55e";

        ctx.strokeRect(x, y, w, h);

        const label = `${b.label} ${Number(b.score).toFixed(2)}`;
        const tw = ctx.measureText(label).width + 12;
        const th = Math.max(22, overlay.width / 45);

        ctx.fillRect(x, Math.max(0, y - th), tw, th);
        ctx.fillStyle = "#020617";
        ctx.fillText(label, x + 6, Math.max(0, y - th) + 3);
    });
}

function captureFrame(maxSide = 1280) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    let w = vw;
    let h = vh;

    if (Math.max(w, h) > maxSide) {
        const s = maxSide / Math.max(w, h);
        w = Math.round(w * s);
        h = Math.round(h * s);
    }

    sendCanvas.width = w;
    sendCanvas.height = h;
    sendCtx.drawImage(video, 0, 0, w, h);

    return sendCanvas.toDataURL("image/jpeg", 0.82);
}

videoInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];

    if (!file) {
        setStatus("Chưa chọn video");
        return;
    }

    running = false;
    busy = false;
    lastBoxes = [];
    sessionId = crypto.randomUUID();
    ocrResults.innerHTML = "";
    clearOverlay();

    const url = URL.createObjectURL(file);
    video.src = url;
    video.load();

    video.onloadedmetadata = async () => {
        resizeOverlay();

        await fetch("/reset", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({session_id: sessionId})
        });

        setStatus(`Đã tải: ${file.name} | ${video.videoWidth}x${video.videoHeight}`);
    };

    video.onerror = () => {
        setStatus("Video không phát được. Hãy dùng MP4 H.264 hoặc WebM.");
        alert("Trình duyệt không đọc được video này. Hãy đổi sang MP4 H.264.");
    };
});

async function detectOnce() {
    if (!running || busy || video.paused || video.ended || video.readyState < 2) return;

    const now = performance.now();
    if (now - lastSent < autoDelay) return;

    busy = true;
    lastSent = now;

    try {
        const img = captureFrame(1280);
        const t0 = performance.now();

        const res = await fetch("/detect_frame", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                image: img,
                conf: Number(conf.value),
                imgsz: Number(imgsz.value),
                session_id: sessionId,
                max_top: 5
            })
        });

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();
        const latency = performance.now() - t0;

        latencyEl.textContent = `${Math.round(latency)} ms`;
        autoDelay = Math.min(650, Math.max(120, latency * 1.25));
        delayEl.textContent = `${Math.round(autoDelay)} ms`;
        topCountEl.textContent = data.top_count ?? 0;

        lastBoxes = data.boxes || [];
        drawBoxes(lastBoxes, data.frame_w, data.frame_h);

        if (lastBoxes.length > 0) {
            setStatus(`Đang detect: ${lastBoxes.length} bbox`);
        } else {
            setStatus("Đang detect: chưa thấy biển số");
        }
    } catch (e) {
        setStatus("Lỗi detect");
        console.error(e);
    } finally {
        busy = false;
    }
}

function loop() {
    if (running) {
        detectOnce();
        requestAnimationFrame(loop);
    }
}

startBtn.addEventListener("click", async () => {
    if (!video.src) {
        alert("Hãy upload video trước.");
        return;
    }

    resizeOverlay();
    running = true;
    setStatus("Đang chạy");

    try {
        await video.play();
        requestAnimationFrame(loop);
    } catch (e) {
        setStatus("Không phát được video");
        console.error(e);
    }
});

stopBtn.addEventListener("click", () => {
    running = false;
    video.pause();
    setStatus("Đã dừng");
});

video.addEventListener("play", () => {
    resizeOverlay();
    if (running) requestAnimationFrame(loop);
});

video.addEventListener("pause", () => {
    setStatus("Video pause");
});

video.addEventListener("seeked", () => {
    lastBoxes = [];
    clearOverlay();
});

video.addEventListener("ended", () => {
    running = false;
    setStatus("Video kết thúc");
});

window.addEventListener("resize", () => {
    resizeOverlay();
});

ocrBtn.addEventListener("click", async () => {
    setStatus("Đang OCR top crop...");
    ocrResults.innerHTML = "";

    const res = await fetch("/ocr_top", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({session_id: sessionId})
    });

    const data = await res.json();
    const rows = data.rows || [];

    if (rows.length === 0) {
        ocrResults.innerHTML = "<p>Không có bbox để OCR.</p>";
        setStatus("Không có bbox OCR");
        return;
    }

    rows.forEach(r => {
        const card = document.createElement("div");
        card.className = "ocr-card";

        card.innerHTML = `
            <img src="data:image/jpeg;base64,${r.image}" alt="crop">
            <div class="plate-text">${r.text || "Không đọc được"}</div>
            <div><b>Top ${r.rank}</b> | conf=${r.score}</div>
            <div class="cands">${(r.cands || []).join(", ")}</div>
        `;

        ocrResults.appendChild(card);
    });

    setStatus("OCR xong");
});