import time
import queue
import threading
import cv2
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

import lp_app.config as C
from lp_app.ui import theme as T
from lp_app.ui.widgets import ctk_image_bgr, ResultSidePanel
from lp_app.services.vision import TrackStore, detect_track, crop_with_pad, enhance_crop, crop_quality, draw_boxes, fit_frame
from lp_app.services.ocr_service import plate_key, duplicate


AUTO_OCR_QUALITY = getattr(C, "AUTO_OCR_QUALITY", 0.55)
AUTO_OCR_INTERVAL_SEC = getattr(C, "AUTO_OCR_INTERVAL_SEC", 0.8)


class RealtimeTab(ctk.CTkFrame):
    def __init__(self, parent, model_service, ocr_service, mode="video"):
        super().__init__(parent, fg_color=T.BG)

        self.model_service = model_service
        self.ocr_service = ocr_service
        self.mode = mode

        self.cap = None
        self.model = None
        self.video_path = ""

        self.running = False
        self.paused = False
        self.finish_requested = False
        self.worker = None

        self.frames = queue.Queue(maxsize=1)
        self.tracks = TrackStore()

        self.frame_idx = 0
        self.fps_show = 0
        self.last_crop_ui = 0
        self.last_ocr_check = 0
        self.ocr_busy = False
        self.ocr_records = []
        self.img = None

        self.build()
        self.after(25, self.update_frame)

    def build(self):
        top = ctk.CTkFrame(self, fg_color=T.PANEL, border_width=1, border_color=T.BORDER)
        top.pack(fill="x", padx=12, pady=12)

        if self.mode == "video":
            self.btn_load = ctk.CTkButton(top, text="Tải video", command=self.load_video, **T.button_style())
            self.btn_load.pack(side="left", padx=8, pady=8)

        self.btn_start = ctk.CTkButton(top, text="Bắt đầu", command=self.start, **T.button_style())
        self.btn_stop = ctk.CTkButton(top, text="Dừng", command=self.stop, state="disabled", **T.button_style())
        self.btn_finish = ctk.CTkButton(top, text="Kết thúc", command=self.finish, state="disabled", **T.button_style())

        self.btn_start.pack(side="left", padx=8, pady=8)
        self.btn_stop.pack(side="left", padx=8, pady=8)
        self.btn_finish.pack(side="left", padx=8, pady=8)

        self.status = ctk.CTkLabel(top, text=self.default_status(), font=T.FONT_BOLD, text_color=T.TEXT)
        self.status.pack(side="left", padx=20)

        main = ctk.CTkFrame(self, fg_color=T.BG)
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0, minsize=260)
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(main, fg_color=T.PANEL, border_width=1, border_color=T.BORDER)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        title = "Video phát hiện biển số" if self.mode == "video" else "Webcam phát hiện biển số"
        ctk.CTkLabel(left, text=title, font=T.TITLE, text_color=T.TEXT).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        self.video_label = tk.Label(left, text="", bg="#F0F0F0")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

        self.side = ResultSidePanel(main)
        self.side.grid(row=0, column=1, sticky="ns", padx=(8, 0))

        if self.mode == "video":
            self.btn_start.configure(state="disabled")

    def default_status(self):
        return "Tab 2: chưa tải video." if self.mode == "video" else "Tab 3: sẵn sàng bật webcam."

    def is_busy(self):
        return self.running or self.paused

    def clear_runtime_state(self, keep_video=False):
        self.running = False
        self.paused = False
        self.finish_requested = False

        self.frame_idx = 0
        self.fps_show = 0
        self.last_crop_ui = 0
        self.last_ocr_check = 0
        self.ocr_busy = False
        self.ocr_records = []

        self.tracks.clear()
        self.side.clear()

        try:
            while True:
                self.frames.get_nowait()
        except queue.Empty:
            pass

        if not keep_video:
            self.img = None
            try:
                self.video_label.configure(image="", text="")
            except Exception:
                pass

    def load_video(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if not path:
            self.status.configure(text="Tab 2: đã hủy tải video.")
            return

        self.stop_worker()
        self.clear_runtime_state()

        self.video_path = path
        self.cap = cv2.VideoCapture(path)

        ok, frame = self.cap.read()
        if ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.show_frame(fit_frame(frame, C.VIDEO_MAX_W, C.VIDEO_MAX_H))

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_finish.configure(state="normal")
        self.status.configure(text="Tab 2: phiên video mới đã reset kết quả cũ. Bấm Bắt đầu.")

    def start(self):
        if self.mode == "webcam":
            self.stop_worker()
            self.clear_runtime_state()
            self.cap = cv2.VideoCapture(0)
            self.status.configure(text="Tab 3: phiên webcam mới đã reset kết quả cũ.")
        elif self.cap is None:
            return
        else:
            self.clear_runtime_state(keep_video=True)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.model_service.reset_tracker()
        self.model = self.model_service.load()

        self.running = True
        self.paused = False
        self.finish_requested = False

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_finish.configure(state="normal")

        tab = "Tab 2" if self.mode == "video" else "Tab 3"
        self.status.configure(text=f"{tab}: đang detect realtime...")

        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def stop(self):
        self.paused = True
        self.running = False

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_finish.configure(state="normal")

        tab = "Tab 2" if self.mode == "video" else "Tab 3"
        self.status.configure(text=f"{tab}: đã dừng tạm thời.")

    def finish(self):
        self.finish_requested = True
        self.running = False
        self.paused = False

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_finish.configure(state="normal")

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        tab = "Tab 2" if self.mode == "video" else "Tab 3"
        self.status.configure(text=f"{tab}: đã kết thúc phiên. OCR realtime được {len(self.ocr_records)} biển số.")

    def stop_worker(self):
        self.finish_requested = True
        self.running = False
        self.paused = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def loop(self):
        last = time.time()

        while not self.finish_requested:
            if self.paused or not self.running:
                time.sleep(0.03)
                continue

            ok, frame = self.cap.read()

            if not ok:
                self.finish_requested = True
                self.running = False

                tab = "Tab 2" if self.mode == "video" else "Tab 3"
                self.after(0, lambda: self.status.configure(text=f"{tab}: video/webcam đã kết thúc."))
                self.after(0, lambda: self.btn_start.configure(state="normal"))
                self.after(0, lambda: self.btn_stop.configure(state="disabled"))

                if not self.ocr_records:
                    threading.Thread(target=self.finalize, daemon=True).start()

                break

            h, w = frame.shape[:2]
            boxes = detect_track(self.model, frame, C.CONF, C.IMGSZ, C.DEVICE, C.HALF, C.TRACKER, self.frame_idx)

            for b in boxes:
                raw_crop = crop_with_pad(frame, b["box"])
                crop = enhance_crop(raw_crop)
                q = crop_quality(crop, b["conf"], b["box"], w, h)
                self.tracks.add(b["track_id"], crop, q, b["conf"], b["box"], self.frame_idx)

            self.frame_idx += 1

            now = time.time()
            fps = 1 / max(now - last, 1e-6)
            last = now
            self.fps_show = 0.9 * self.fps_show + 0.1 * fps if self.fps_show else fps

            show = draw_boxes(frame, boxes, len(self.tracks), self.fps_show)
            show = fit_frame(show, C.VIDEO_MAX_W, C.VIDEO_MAX_H)

            if self.frames.full():
                try:
                    self.frames.get_nowait()
                except Exception:
                    pass

            self.frames.put(show)

            tab = "Tab 2" if self.mode == "video" else "Tab 3"
            self.after(0, lambda n=len(boxes): self.status.configure(text=f"{tab}: đang detect realtime | bbox hiện tại: {n}"))

            if now - self.last_crop_ui >= C.REALTIME_CROP_UPDATE_SEC:
                self.last_crop_ui = now
                self.after(0, self.side.show_crops, self.tracks.snapshot(limit=1))

            if now - self.last_ocr_check >= AUTO_OCR_INTERVAL_SEC:
                self.last_ocr_check = now
                self.auto_ocr_best_crop()

    def auto_ocr_best_crop(self):
        if self.ocr_busy:
            return

        best_item = None

        for tid, t in self.tracks.tracks.items():
            if not t["candidates"] or t["hits"] < 2:
                continue

            best = max(t["candidates"], key=lambda x: x["quality"])
            if best["quality"] < AUTO_OCR_QUALITY:
                continue

            if best_item is None or best["quality"] > best_item["quality"]:
                best_item = {"track_id": tid, "track": t, "quality": best["quality"]}

        if best_item is None:
            return

        self.ocr_busy = True
        threading.Thread(target=self.ocr_worker, args=(best_item,), daemon=True).start()

    def ocr_worker(self, best_item):
        try:
            t = best_item["track"]
            text, crop, quality = self.ocr_service.best_text(t["candidates"])
            key = plate_key(text)

            if len(key) < 4:
                return

            for old in self.ocr_records:
                if duplicate(text, old["text"]):
                    return

            record = {
                "track_id": best_item["track_id"],
                "text": text,
                "crop": crop,
                "hits": t["hits"],
                "first_frame": t["first_frame"],
                "conf": t["best_conf"],
                "quality": quality,
            }

            self.ocr_records.append(record)
            self.after(0, self.side.show_ocr, self.ocr_records)

            tab = "Tab 2" if self.mode == "video" else "Tab 3"
            self.after(0, lambda: self.status.configure(text=f"{tab}: OCR realtime được {len(self.ocr_records)} biển số."))

        finally:
            self.ocr_busy = False

    def update_frame(self):
        try:
            frame = self.frames.get_nowait()
            self.show_frame(frame)
        except queue.Empty:
            pass

        self.after(20, self.update_frame)

    def show_frame(self, frame):
        label_w = max(self.video_label.winfo_width(), C.VIDEO_MAX_W)
        label_h = max(self.video_label.winfo_height(), C.VIDEO_MAX_H)

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return

        scale = min(label_w / max(w, 1), label_h / max(h, 1))

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        self.img = ImageTk.PhotoImage(pil)
        self.video_label.configure(image=self.img, text="")
        self.video_label.image = self.img

    def finalize(self):
        records = self.ocr_service.records_from_tracks(self.tracks.tracks)

        if self.ocr_records:
            old_texts = [r["text"] for r in self.ocr_records]
            for r in records:
                if not any(duplicate(r["text"], old) for old in old_texts):
                    self.ocr_records.append(r)
        else:
            self.ocr_records = records

        self.after(0, self.side.show_ocr, self.ocr_records)

        tab = "Tab 2" if self.mode == "video" else "Tab 3"
        self.after(0, lambda: self.status.configure(text=f"{tab}: hoàn tất OCR. Tổng {len(self.ocr_records)} biển số."))

    def destroy(self):
        self.stop_worker()
        super().destroy()

