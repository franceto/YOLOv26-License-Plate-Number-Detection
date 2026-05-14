const videoInput = document.getElementById("videoInput");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const finishBtn = document.getElementById("finishBtn");
const statusText = document.getElementById("statusText");
const totalText = document.getElementById("totalText");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const resultsBox = document.getElementById("results");

let sessionId = null;
let running = false;
let busy = false;
let lastSent = 0;
let autoDelay = 130;
let lastBoxes = [];
let lastFrameW = 1;
let lastFrameH = 1;

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

async function createSession() {
    const res = await fetch("/api/new_session");
    const data = await res.json();
    sessionId = data.session_id;
}

function resizeOverlay() {
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;

    overlay.width = Math.round(rect.width * dpr);
    overlay.height = Math.round(rect.height * dpr);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawBoxes() {
    resizeOverlay();

    const rect = video.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    const sx = rect.width / lastFrameW;
    const sy = rect.height / lastFrameH;

    for (const item of lastBoxes) {
        const [x1, y1, x2, y2] = item.box;

        const rx = x1 * sx;
        const ry = y1 * sy;
        const rw = (x2 - x1) * sx;
        const rh = (y2 - y1) * sy;

        ctx.lineWidth = 3;
        ctx.strokeStyle = "#00ff66";
        ctx.strokeRect(rx, ry, rw, rh);

        const label = `ID ${item.track_id} | ${Number(item.conf).toFixed(2)}`;
        ctx.font = "16px Arial";
        const tw = ctx.measureText(label).width + 12;

        ctx.fillStyle = "#00ff66";
        ctx.fillRect(rx, Math.max(0, ry - 26), tw, 24);

        ctx.fillStyle = "#0f172a";
        ctx.fillText(label, rx + 6, Math.max(17, ry - 8));
    }

    requestAnimationFrame(drawBoxes);
}

function captureFrame() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;

    if (!vw || !vh) return null;

    const maxSide = 960;
    let sw = vw;
    let sh = vh;

    if (Math.max(vw, vh) > maxSide) {
        const scale = maxSide / Math.max(vw, vh);
        sw = Math.round(vw * scale);
        sh = Math.round(vh * scale);
    }

    captureCanvas.width = sw;
    captureCanvas.height = sh;
    captureCtx.drawImage(video, 0, 0, sw, sh);

    return captureCanvas.toDataURL("image/jpeg", 0.82);
}

async function detectLoop(now) {
    if (!running) return;

    if (!video.paused && !video.ended && !busy && now - lastSent >= autoDelay) {
        const image = captureFrame();

        if (image) {
            busy = true;
            lastSent = now;

            const t0 = performance.now();

            try {
                const res = await fetch("/api/detect_frame", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        image: image,
                        conf: 0.18,
                        imgsz: 640
                    })
                });

                const data = await res.json();

                lastBoxes = data.boxes || [];
                lastFrameW = data.frame_w || 1;
                lastFrameH = data.frame_h || 1;
                totalText.textContent = data.total_tracks || 0;

                const latency = performance.now() - t0;
                autoDelay = Math.max(120, Math.min(360, latency * 1.2));

                statusText.textContent = `Đang detect realtime | bbox hiện tại: ${lastBoxes.length}`;
            } catch (err) {
                statusText.textContent = "Lỗi detect frame";
            }

            busy = false;
        }
    }

    requestAnimationFrame(detectLoop);
}

function renderResults(items) {
    resultsBox.innerHTML = "";

    if (!items.length) {
        resultsBox.innerHTML = "<p>Chưa có kết quả OCR.</p>";
        return;
    }

    for (const item of items) {
        const card = document.createElement("div");
        card.className = "resultCard";

        const img = document.createElement("img");
        img.src = item.image;
        img.alt = "plate crop";

        const text = document.createElement("div");
        text.className = "plateText";
        text.textContent = item.text || "OCR chưa đọc rõ";

        const meta1 = document.createElement("div");
        meta1.className = "meta";
        meta1.textContent = `Track ID: ${item.track_id}`;

        const meta2 = document.createElement("div");
        meta2.className = "meta";
        meta2.textContent = `Số frame thấy biển: ${item.hits}`;

        const meta3 = document.createElement("div");
        meta3.className = "meta";
        meta3.textContent = `Conf: ${item.conf} | Quality: ${item.quality}`;

        card.appendChild(img);
        card.appendChild(text);
        card.appendChild(meta1);
        card.appendChild(meta2);
        card.appendChild(meta3);

        resultsBox.appendChild(card);
    }
}

async function finishOcr() {
    if (!sessionId) return;

    statusText.textContent = "Đang OCR crop tốt nhất...";

    const res = await fetch("/api/ocr_best", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            session_id: sessionId
        })
    });

    const data = await res.json();

    totalText.textContent = data.total || 0;
    renderResults(data.results || []);
    statusText.textContent = `Hoàn tất. Lấy ${data.total || 0} biển số tốt nhất, không trùng lặp.`;
}

videoInput.addEventListener("change", async () => {
    const file = videoInput.files[0];
    if (!file) return;

    await createSession();

    video.src = URL.createObjectURL(file);
    video.load();

    running = false;
    busy = false;
    lastBoxes = [];
    totalText.textContent = "0";
    statusText.textContent = "Đã tải video. Bấm Bắt đầu để phát và detect realtime.";
    resultsBox.innerHTML = "<p>Chưa có kết quả OCR.</p>";

    startBtn.disabled = false;
    stopBtn.disabled = true;
    finishBtn.disabled = false;
});

startBtn.addEventListener("click", async () => {
    if (!sessionId || !video.src) return;

    running = true;
    busy = false;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    finishBtn.disabled = false;

    statusText.textContent = "Đang phát video và detect realtime...";

    await video.play();

    requestAnimationFrame(detectLoop);
});

stopBtn.addEventListener("click", () => {
    running = false;
    video.pause();

    startBtn.disabled = false;
    stopBtn.disabled = true;
    finishBtn.disabled = false;

    statusText.textContent = "Đã dừng video.";
});

finishBtn.addEventListener("click", async () => {
    running = false;
    video.pause();

    startBtn.disabled = false;
    stopBtn.disabled = true;
    finishBtn.disabled = true;

    await finishOcr();
});

video.addEventListener("ended", async () => {
    running = false;
    stopBtn.disabled = true;
    startBtn.disabled = false;
    finishBtn.disabled = true;

    await finishOcr();
});

video.addEventListener("loadedmetadata", resizeOverlay);
window.addEventListener("resize", resizeOverlay);

requestAnimationFrame(drawBoxes);