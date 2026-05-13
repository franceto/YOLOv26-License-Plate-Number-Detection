from pathlib import Path
import argparse
import difflib
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

try:
    from streamlit_app.src.ocr_engine import recognize_plate, recognize_plate_fast
except Exception:
    recognize_plate = None
    recognize_plate_fast = None


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "weights" / "yolo26_bienso_best.pt"

TOP_K_CROPS = 5
MIN_TRACK_HITS = 2
REALTIME_CROP_UPDATE_SEC = 0.45


UI = {
    "bg": "#F5F6F8",
    "panel": "#FFFFFF",
    "soft": "#FAFAFA",
    "border": "#D9DEE7",
    "text": "#111111",
    "muted": "#666666",
    "black": "#000000",
    "red": "#FF0000",
    "white": "#FFFFFF",
}


def clean_text(text):
    if not text:
        return ""
    text = str(text).upper()
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t_]+", "", text)
    text = text.replace("|", "1")
    text = re.sub(r"[^A-Z0-9.\-\n]", "", text)
    return text.strip()


def plate_key(text):
    text = clean_text(text).replace("\n", "")
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_ocr_result(result):
    if result is None:
        return ""

    if isinstance(result, str):
        return clean_text(result)

    if isinstance(result, dict):
        for key in ["text", "plate", "best_text", "final_text", "result"]:
            if result.get(key):
                return clean_text(result[key])
        return ""

    if isinstance(result, tuple):
        return normalize_ocr_result(result[0])

    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                for key in ["text", "plate", "best_text"]:
                    if item.get(key):
                        texts.append(str(item[key]))
                        break
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if isinstance(item[1], str):
                    texts.append(item[1])
        return clean_text("".join(texts))

    return clean_text(str(result))


class OCREngine:
    def __init__(self):
        self.ocr = None

    def get_ocr(self):
        if self.ocr is None:
            if RapidOCR is None:
                raise RuntimeError("Chưa cài rapidocr-onnxruntime")
            self.ocr = RapidOCR()
        return self.ocr

    def raw_rapidocr(self, crop):
        try:
            return normalize_ocr_result(self.get_ocr()(crop))
        except Exception:
            return ""

    def project_ocr(self, crop):
        for fn in [recognize_plate, recognize_plate_fast]:
            if fn is None:
                continue

            try:
                return normalize_ocr_result(fn(crop, self.get_ocr()))
            except TypeError:
                try:
                    return normalize_ocr_result(fn(crop))
                except Exception:
                    pass
            except Exception:
                pass

        return self.raw_rapidocr(crop)

    def candidates(self, crop):
        if crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        ratio = w / max(h, 1)
        texts = []

        full = self.project_ocr(crop)
        if full:
            texts.append(full)

        if ratio < 3.2 and h >= 42:
            mid = h // 2
            top = self.raw_rapidocr(crop[:mid, :])
            bot = self.raw_rapidocr(crop[mid:, :])
            if top and bot:
                texts.append(clean_text(top + "\n" + bot))

        if ratio < 1.9 and h >= 72:
            a = h // 3
            b = 2 * h // 3
            lines = [
                self.raw_rapidocr(crop[:a, :]),
                self.raw_rapidocr(crop[a:b, :]),
                self.raw_rapidocr(crop[b:, :]),
            ]
            lines = [x for x in lines if x]
            if len(lines) >= 2:
                texts.append(clean_text("\n".join(lines)))

        out = []
        seen = set()

        for t in texts:
            k = plate_key(t)
            if len(k) >= 4 and k not in seen:
                out.append(t)
                seen.add(k)

        return out


def clamp_box(box, w, h):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return [int(x1), int(y1), int(x2), int(y2)]


def valid_plate_box(box, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    if bw < 10 or bh < 8:
        return False

    ratio = bw / max(bh, 1)
    area = bw * bh

    if ratio < 0.55 or ratio > 8.5:
        return False
    if area < 100:
        return False
    if area / max(w * h, 1) < 0.00002:
        return False

    return True


def crop_with_pad(frame, box, pad_ratio=0.20):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w - 1, x2 + px)
    y2 = min(h - 1, y2 + py)

    return frame[y1:y2, x1:x2].copy()


def enhance_crop_for_ocr(crop, min_h=96, max_scale=4.0):
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop

    scale = min(max_scale, max(1.0, min_h / h))

    if scale > 1.0:
        crop = cv2.resize(
            crop,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    blur = cv2.GaussianBlur(crop, (0, 0), 1.0)
    crop = cv2.addWeighted(crop, 1.35, blur, -0.35, 0)

    return crop


def crop_quality(crop, conf, box, frame_w, frame_h):
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())

    x1, y1, x2, y2 = box
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)

    sharp_score = min(sharpness / 420.0, 1.0)
    bright_score = max(0.0, 1.0 - abs(brightness - 135.0) / 135.0)
    area_score = min(area_ratio * 130.0, 1.0)

    return 0.52 * float(conf) + 0.28 * sharp_score + 0.15 * bright_score + 0.05 * area_score


def add_candidate(tracks, track_id, crop, quality, conf, box, frame_idx):
    if track_id not in tracks:
        tracks[track_id] = {
            "track_id": int(track_id),
            "hits": 0,
            "first_frame": frame_idx,
            "last_frame": frame_idx,
            "best_conf": 0.0,
            "candidates": [],
        }

    t = tracks[track_id]
    t["hits"] += 1
    t["last_frame"] = frame_idx
    t["best_conf"] = max(t["best_conf"], float(conf))

    t["candidates"].append(
        {
            "crop": crop,
            "quality": float(quality),
            "conf": float(conf),
            "box": box,
            "frame": frame_idx,
        }
    )

    t["candidates"] = sorted(
        t["candidates"],
        key=lambda x: x["quality"],
        reverse=True,
    )[:TOP_K_CROPS]


def choose_best_text(ocr_engine, candidates):
    votes = {}

    for item in candidates:
        crop = item["crop"]
        quality = item["quality"]

        for text in ocr_engine.candidates(crop):
            key = plate_key(text)

            if len(key) < 4:
                continue

            score = quality + min(len(key), 10) * 0.03

            if "\n" in text:
                score += 0.08

            if key not in votes:
                votes[key] = {
                    "text": text,
                    "score": 0.0,
                    "count": 0,
                    "crop": crop,
                    "quality": quality,
                }

            votes[key]["score"] += score
            votes[key]["count"] += 1

            if quality > votes[key]["quality"]:
                votes[key]["crop"] = crop
                votes[key]["quality"] = quality
                votes[key]["text"] = text

    if not votes:
        best = candidates[0]
        return "", best["crop"], best["quality"]

    best = max(votes.values(), key=lambda x: (x["count"], x["score"], x["quality"]))
    return best["text"], best["crop"], best["quality"]


def is_duplicate(a, b):
    ka = plate_key(a)
    kb = plate_key(b)

    if len(ka) < 4 or len(kb) < 4:
        return False

    if ka == kb:
        return True

    return difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.86


def deduplicate(records):
    records = sorted(records, key=lambda x: x["quality"], reverse=True)
    kept = []

    for r in records:
        if any(is_duplicate(r["text"], k["text"]) for k in kept):
            continue
        kept.append(r)

    return sorted(kept, key=lambda x: x["first_frame"])


def fit_frame(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)

    if scale >= 1.0:
        return frame

    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_boxes(frame, boxes, total_tracks, fps_show):
    out = frame.copy()

    for item in boxes:
        x1, y1, x2, y2 = item["box"]
        tid = item["track_id"]
        conf = item["conf"]

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)

        label = f"ID {tid} | {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        y0 = max(0, y1 - th - 10)

        cv2.rectangle(out, (x1, y0), (x1 + tw + 10, y1), (0, 255, 80), -1)
        cv2.putText(out, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

    info = f"tracks: {total_tracks} | fps: {fps_show:.1f}"
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 340), 34), (0, 0, 0), -1)
    cv2.putText(out, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    return out


def bgr_to_tk(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)


class PlateDemoApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args

        self.root.title("YOLOv26 License Plate Realtime Demo")
        self.root.geometry("1320x860")
        self.root.configure(bg=UI["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.model = None
        self.cap = None
        self.video_path = ""
        self.running = False
        self.paused = False
        self.finish_requested = False
        self.worker = None
        self.frame_queue = queue.Queue(maxsize=1)

        self.tracks = {}
        self.frame_idx = 0
        self.fps_show = 0.0
        self.source_fps = 25
        self.delay = 1
        self.last_crop_ui_update = 0.0

        self.status_var = tk.StringVar(value="Chưa tải video")
        self.total_var = tk.StringVar(value="0")

        self.buttons = {}
        self.crop_images = []
        self.ocr_images = []

        self.build_ui()
        self.root.after(30, self.update_frame)

    def make_panel(self, parent, **kwargs):
        return tk.Frame(
            parent,
            bg=UI["panel"],
            highlightbackground=UI["border"],
            highlightthickness=1,
            bd=0,
            **kwargs,
        )

    def build_ui(self):
        self.build_top_bar()
        self.build_status_bar()
        self.build_main_area()

    def build_top_bar(self):
        top = self.make_panel(self.root)
        top.pack(fill="x", padx=14, pady=(12, 8))

        button_wrap = tk.Frame(top, bg=UI["panel"])
        button_wrap.pack(anchor="w", padx=12, pady=8)

        self.buttons["load"] = self.create_button(button_wrap, "Tải video", self.load_video)
        self.buttons["start"] = self.create_button(button_wrap, "Bắt đầu", self.start)
        self.buttons["stop"] = self.create_button(button_wrap, "Dừng", self.stop)
        self.buttons["finish"] = self.create_button(button_wrap, "Kết thúc", self.finish)

        self.buttons["load"].grid(row=0, column=0, padx=(0, 8))
        self.buttons["start"].grid(row=0, column=1, padx=8)
        self.buttons["stop"].grid(row=0, column=2, padx=8)
        self.buttons["finish"].grid(row=0, column=3, padx=8)

        self.set_button_state(loaded=False, running=False)
        self.activate_button(None)

    def create_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 9, "bold"),
            width=13,
            height=1,
            bd=1,
            relief="groove",
            bg=UI["white"],
            fg=UI["black"],
            activebackground=UI["black"],
            activeforeground=UI["red"],
            cursor="hand2",
            takefocus=False,
            padx=6,
            pady=4,
        )

    def activate_button(self, active_key):
        for key, btn in self.buttons.items():
            if key == active_key:
                btn.config(bg=UI["black"], fg=UI["red"])
            else:
                btn.config(bg=UI["white"], fg=UI["black"])

    def build_status_bar(self):
        status = self.make_panel(self.root)
        status.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(
            status,
            text="Trạng thái:",
            font=("Segoe UI", 10),
            bg=UI["panel"],
            fg=UI["muted"],
        ).pack(side="left", padx=(14, 4), pady=10)

        tk.Label(
            status,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
            bg=UI["panel"],
            fg=UI["text"],
        ).pack(side="left", pady=10)

        tk.Label(
            status,
            text="Tổng số biển số tốt nhất:",
            font=("Segoe UI", 10),
            bg=UI["panel"],
            fg=UI["muted"],
        ).pack(side="left", padx=(30, 4), pady=10)

        tk.Label(
            status,
            textvariable=self.total_var,
            font=("Segoe UI", 10, "bold"),
            bg=UI["panel"],
            fg=UI["text"],
        ).pack(side="left", pady=10)

    def build_main_area(self):
        main = tk.Frame(self.root, bg=UI["bg"])
        main.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        main.grid_columnconfigure(0, weight=5)
        main.grid_columnconfigure(1, weight=0)
        main.grid_rowconfigure(0, weight=1)

        self.build_video_panel(main)
        self.build_right_tabs(main)

    def build_video_panel(self, parent):
        video_panel = self.make_panel(parent)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_panel.grid_rowconfigure(1, weight=1)
        video_panel.grid_columnconfigure(0, weight=1)

        tk.Label(
            video_panel,
            text="Video phát hiện biển số",
            font=("Segoe UI", 16, "bold"),
            bg=UI["panel"],
            fg=UI["text"],
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        video_border = tk.Frame(
            video_panel,
            bg=UI["white"],
            highlightbackground=UI["border"],
            highlightthickness=1,
            bd=0,
        )
        video_border.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        video_border.grid_rowconfigure(0, weight=1)
        video_border.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_border, bg="#F0F0F0")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def build_right_tabs(self, parent):
        right = self.make_panel(parent, width=360)
        right.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        right.grid_propagate(False)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        tk.Label(
            right,
            text="Kết quả",
            font=("Segoe UI", 16, "bold"),
            bg=UI["panel"],
            fg=UI["text"],
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 8))

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Clean.TNotebook",
            background=UI["panel"],
            borderwidth=0,
        )
        style.configure(
            "Clean.TNotebook.Tab",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 7),
            background=UI["white"],
            foreground=UI["black"],
        )
        style.map(
            "Clean.TNotebook.Tab",
            background=[("selected", UI["black"])],
            foreground=[("selected", UI["red"])],
        )

        self.tabs = ttk.Notebook(right, style="Clean.TNotebook")
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.crop_tab = tk.Frame(self.tabs, bg=UI["soft"])
        self.ocr_tab = tk.Frame(self.tabs, bg=UI["soft"])

        self.tabs.add(self.crop_tab, text="Ảnh crop realtime")
        self.tabs.add(self.ocr_tab, text="OCR")

        self.crop_canvas, self.crop_frame = self.create_scroll_area(self.crop_tab)
        self.ocr_canvas, self.ocr_frame = self.create_scroll_area(self.ocr_tab)

        self.clear_crop_tab()
        self.clear_ocr_tab()

    def create_scroll_area(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            parent,
            bg=UI["soft"],
            highlightbackground=UI["border"],
            highlightthickness=1,
            yscrollincrement=20,
        )
        canvas.grid(row=0, column=0, sticky="nsew")

        y_scroll = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")

        canvas.configure(yscrollcommand=y_scroll.set)

        inner = tk.Frame(canvas, bg=UI["soft"])
        canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.bind("<Enter>", lambda e, c=canvas: c.bind_all("<MouseWheel>", lambda ev: c.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
        canvas.bind("<Leave>", lambda e, c=canvas: c.unbind_all("<MouseWheel>"))

        return canvas, inner

    def set_button_state(self, loaded=False, running=False):
        self.buttons["start"].config(state="normal" if loaded and not running else "disabled")
        self.buttons["stop"].config(state="normal" if running else "disabled")
        self.buttons["finish"].config(state="normal" if loaded else "disabled")

    def load_model(self):
        if self.model is not None:
            return

        model_path = Path(self.args.model)

        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

        self.status_var.set("Đang load YOLOv26...")
        self.root.update_idletasks()

        self.model = YOLO(str(model_path))

        try:
            self.model.predictor = None
        except Exception:
            pass

    def load_video(self):
        self.activate_button("load")

        path = filedialog.askopenfilename(
            title="Chọn video biển số",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            self.activate_button(None)
            return

        self.stop_worker(join=True)
        self.reset_state()

        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được video")
            self.video_path = ""
            self.activate_button(None)
            return

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.delay = 1 if self.args.fast else max(1, int(1000 / self.source_fps))

        ok, frame = self.cap.read()

        if ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            show = fit_frame(frame, self.args.max_width, self.args.max_height)
            tk_img = bgr_to_tk(show)
            self.video_label.configure(image=tk_img)
            self.video_label.image = tk_img

        self.status_var.set("Đã tải video. Bấm Bắt đầu để detect realtime.")
        self.set_button_state(loaded=True, running=False)

    def reset_state(self):
        self.running = False
        self.paused = False
        self.finish_requested = False
        self.tracks = {}
        self.frame_idx = 0
        self.fps_show = 0.0
        self.last_crop_ui_update = 0.0
        self.total_var.set("0")
        self.clear_crop_tab()
        self.clear_ocr_tab()

        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

    def start(self):
        if not self.video_path:
            return

        self.activate_button("start")

        try:
            self.load_model()
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            self.activate_button(None)
            return

        if self.worker and self.worker.is_alive():
            self.running = True
            self.paused = False
            self.status_var.set("Đang detect realtime...")
            self.set_button_state(loaded=True, running=True)
            return

        self.running = True
        self.paused = False
        self.finish_requested = False
        self.status_var.set("Đang detect realtime...")
        self.set_button_state(loaded=True, running=True)

        self.worker = threading.Thread(target=self.video_loop, daemon=True)
        self.worker.start()

    def stop(self):
        self.activate_button("stop")
        self.paused = True
        self.running = False
        self.status_var.set("Đã dừng tạm thời. Bấm Bắt đầu để chạy tiếp.")
        self.set_button_state(loaded=True, running=False)

    def finish(self):
        self.activate_button("finish")
        self.finish_requested = True
        self.running = False
        self.paused = False
        self.set_button_state(loaded=False, running=False)
        self.status_var.set("Đang kết thúc và OCR crop tốt nhất...")
        threading.Thread(target=self.finalize_ocr, daemon=True).start()

    def stop_worker(self, join=False):
        self.finish_requested = True
        self.running = False

        if join and self.worker and self.worker.is_alive():
            self.worker.join(timeout=1.5)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def video_loop(self):
        last_t = time.time()

        while not self.finish_requested:
            if self.paused or not self.running:
                time.sleep(0.03)
                continue

            ok, frame = self.cap.read()

            if not ok:
                self.root.after(0, self.status_var.set, "Video kết thúc. Đang OCR crop tốt nhất...")
                self.root.after(0, self.set_button_state, False, False)
                self.finish_requested = True
                threading.Thread(target=self.finalize_ocr, daemon=True).start()
                break

            h, w = frame.shape[:2]

            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=self.args.tracker,
                    conf=self.args.conf,
                    imgsz=self.args.imgsz,
                    device=self.args.device,
                    half=self.args.half,
                    verbose=False,
                )
            except Exception:
                results = self.model.predict(
                    frame,
                    conf=self.args.conf,
                    imgsz=self.args.imgsz,
                    device=self.args.device,
                    half=self.args.half,
                    verbose=False,
                )

            boxes_now = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.arange(1, len(xyxy) + 1) + self.frame_idx * 10000

                for box_raw, conf, tid in zip(xyxy, confs, ids):
                    box = clamp_box(box_raw, w, h)

                    if not valid_plate_box(box, w, h):
                        continue

                    raw_crop = crop_with_pad(frame, box)
                    ocr_crop = enhance_crop_for_ocr(raw_crop, min_h=self.args.crop_min_height)
                    q = crop_quality(ocr_crop, conf, box, w, h)

                    add_candidate(self.tracks, int(tid), ocr_crop, q, conf, box, self.frame_idx)

                    boxes_now.append(
                        {
                            "track_id": int(tid),
                            "conf": float(conf),
                            "box": box,
                        }
                    )

            self.frame_idx += 1

            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = 1 / max(dt, 1e-6)
            self.fps_show = 0.9 * self.fps_show + 0.1 * inst_fps if self.fps_show else inst_fps

            annotated = draw_boxes(frame, boxes_now, len(self.tracks), self.fps_show)
            annotated = fit_frame(annotated, self.args.max_width, self.args.max_height)

            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(annotated)
            except queue.Full:
                pass

            self.root.after(0, self.total_var.set, str(len(self.tracks)))
            self.root.after(0, self.status_var.set, f"Đang detect realtime | bbox hiện tại: {len(boxes_now)}")

            if now - self.last_crop_ui_update >= REALTIME_CROP_UPDATE_SEC:
                self.last_crop_ui_update = now
                snapshot = self.get_best_crop_snapshot()
                self.root.after(0, self.show_realtime_crops, snapshot)

            if self.delay > 1:
                time.sleep(self.delay / 1000)

    def get_best_crop_snapshot(self):
        items = []

        for tid, t in sorted(self.tracks.items(), key=lambda x: x[1]["first_frame"]):
            if not t["candidates"]:
                continue

            best = max(t["candidates"], key=lambda x: x["quality"])
            items.append(
                {
                    "track_id": tid,
                    "crop": best["crop"].copy(),
                    "quality": best["quality"],
                }
            )

        return items

    def update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            tk_img = bgr_to_tk(frame)
            self.video_label.configure(image=tk_img)
            self.video_label.image = tk_img
        except queue.Empty:
            pass

        self.root.after(20, self.update_frame)

    def clear_crop_tab(self):
        self.crop_images.clear()

        for child in self.crop_frame.winfo_children():
            child.destroy()

        tk.Label(
            self.crop_frame,
            text="Chưa có ảnh crop realtime.",
            bg=UI["soft"],
            fg=UI["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=10, pady=10)

    def clear_ocr_tab(self):
        self.ocr_images.clear()

        for child in self.ocr_frame.winfo_children():
            child.destroy()

        tk.Label(
            self.ocr_frame,
            text="Chưa có kết quả OCR.",
            bg=UI["soft"],
            fg=UI["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=10, pady=10)

    def show_realtime_crops(self, items):
        for child in self.crop_frame.winfo_children():
            child.destroy()

        self.crop_images.clear()

        if not items:
            tk.Label(
                self.crop_frame,
                text="Chưa có ảnh crop realtime.",
                bg=UI["soft"],
                fg=UI["muted"],
                font=("Segoe UI", 10),
            ).pack(anchor="w", padx=10, pady=10)
            return

        for item in items:
            card = tk.Frame(
                self.crop_frame,
                bg=UI["white"],
                highlightbackground=UI["border"],
                highlightthickness=1,
                bd=0,
            )
            card.pack(fill="x", padx=8, pady=8)

            crop_show = fit_frame(item["crop"], 310, 150)
            tk_img = bgr_to_tk(crop_show)
            self.crop_images.append(tk_img)

            img_label = tk.Label(card, image=tk_img, bg="#F2F2F2")
            img_label.pack(padx=8, pady=8)

    def finalize_ocr(self):
        records = []
        ocr_engine = OCREngine()

        for tid, t in sorted(self.tracks.items(), key=lambda x: x[1]["first_frame"]):
            if t["hits"] < MIN_TRACK_HITS or not t["candidates"]:
                continue

            text, crop, quality = choose_best_text(ocr_engine, t["candidates"])
            key = plate_key(text)

            if len(key) < 4:
                continue

            records.append(
                {
                    "track_id": tid,
                    "text": text,
                    "crop": crop,
                    "hits": t["hits"],
                    "first_frame": t["first_frame"],
                    "last_frame": t["last_frame"],
                    "conf": t["best_conf"],
                    "quality": quality,
                }
            )

        records = deduplicate(records)
        self.root.after(0, self.show_results, records)

    def show_results(self, records):
        self.show_ocr_results(records)
        self.total_var.set(str(len(records)))

        if not records:
            self.status_var.set("Hoàn tất. Chưa có OCR đáng tin cậy.")
            return

        self.status_var.set(f"Hoàn tất. Lấy {len(records)} biển số tốt nhất, không trùng lặp.")

    def show_ocr_results(self, records):
        for child in self.ocr_frame.winfo_children():
            child.destroy()

        self.ocr_images.clear()

        if not records:
            tk.Label(
                self.ocr_frame,
                text="Chưa có kết quả OCR đáng tin cậy.",
                bg=UI["soft"],
                fg=UI["red"],
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w", padx=10, pady=10)
            return

        for r in records:
            card = tk.Frame(
                self.ocr_frame,
                bg=UI["white"],
                highlightbackground=UI["border"],
                highlightthickness=1,
                bd=0,
            )
            card.pack(fill="x", padx=8, pady=8)

            crop_show = fit_frame(r["crop"], 310, 140)
            tk_img = bgr_to_tk(crop_show)
            self.ocr_images.append(tk_img)

            img_label = tk.Label(card, image=tk_img, bg="#F2F2F2")
            img_label.pack(padx=8, pady=(8, 4))

            tk.Label(
                card,
                text=r["text"] if r["text"] else "OCR_UNCLEAR",
                bg=UI["white"],
                fg=UI["black"],
                font=("Segoe UI", 16, "bold"),
                justify="center",
            ).pack(padx=8, pady=(0, 10))

    def on_close(self):
        self.stop_worker(join=False)
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--max-width", type=int, default=900)
    parser.add_argument("--max-height", type=int, default=600)
    parser.add_argument("--crop-min-height", type=int, default=96)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = PlateDemoApp(root, args)
    root.mainloop()
