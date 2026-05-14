import tempfile
import zipfile
from pathlib import Path
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox

from lp_app.config import IMAGE_EXTS, ARCHIVE_EXTS, CONF, IMGSZ, DEVICE, HALF
from lp_app.ui import theme as T
from lp_app.ui.widgets import ImageBox, ResultSidePanel
from lp_app.services.vision import (
    imread_unicode,
    detect_predict,
    crop_with_pad,
    enhance_crop,
    crop_quality,
    draw_boxes,
)
from lp_app.services.report_pdf import export_pdf


class ImageTab(ctk.CTkFrame):
    def __init__(self, parent, model_service, ocr_service):
        super().__init__(parent, fg_color=T.BG)

        self.model_service = model_service
        self.ocr_service = ocr_service

        self.paths = []
        self.tempdirs = []
        self.samples = []
        self.batch = False
        self.busy = False

        self.build()

    def build(self):
        top = ctk.CTkFrame(
            self,
            fg_color=T.PANEL,
            border_width=1,
            border_color=T.BORDER,
        )
        top.pack(fill="x", padx=12, pady=12)

        self.btn_load = ctk.CTkButton(
            top,
            text="Tải ảnh",
            command=self.load_one,
            **T.button_style(),
        )

        self.btn_batch = ctk.CTkButton(
            top,
            text="Tải ảnh hàng loạt / zip / rar",
            command=self.load_batch,
            **T.button_style(),
        )

        self.btn_detect = ctk.CTkButton(
            top,
            text="Detect",
            command=self.detect,
            state="disabled",
            **T.button_style(),
        )

        self.btn_pdf = ctk.CTkButton(
            top,
            text="Xuất báo cáo PDF",
            command=self.export_report,
            state="disabled",
            **T.button_style(),
        )

        self.btn_load.pack(side="left", padx=8, pady=8)
        self.btn_batch.pack(side="left", padx=8, pady=8)
        self.btn_detect.pack(side="left", padx=8, pady=8)

        self.status = ctk.CTkLabel(
            top,
            text="Tab 1: chưa tải ảnh.",
            font=T.FONT_BOLD,
            text_color=T.TEXT,
        )
        self.status.pack(side="left", padx=20)

        main = ctk.CTkFrame(self, fg_color=T.BG)
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0, minsize=260)
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(
            main,
            fg_color=T.PANEL,
            border_width=1,
            border_color=T.BORDER,
        )
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left,
            text="Ảnh phát hiện biển số",
            font=T.TITLE,
            text_color=T.TEXT,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        self.original_box = ImageBox(left, "1. Ảnh gốc", h=285)
        self.bbox_box = ImageBox(left, "2. Ảnh bbox vùng biển số", h=285)

        self.original_box.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.bbox_box.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 16))

        self.side = ResultSidePanel(main)
        self.side.grid(row=0, column=1, sticky="ns", padx=(8, 0))

        self.set_pdf_visible(False)

    def is_busy(self):
        return self.busy

    def set_pdf_visible(self, visible):
        if visible and not self.btn_pdf.winfo_ismapped():
            self.btn_pdf.pack(side="left", padx=8, pady=8)

        if not visible and self.btn_pdf.winfo_ismapped():
            self.btn_pdf.pack_forget()

    def clear_outputs(self, keep_original=False):
        self.samples = []
        self.btn_pdf.configure(state="disabled")
        self.set_pdf_visible(False)

        if not keep_original:
            self.original_box.clear("Chưa có ảnh")

        self.bbox_box.clear("Chưa detect")
        self.side.clear()

    def reset_files(self):
        for td in self.tempdirs:
            try:
                td.cleanup()
            except Exception:
                pass

        self.paths = []
        self.tempdirs = []

    def load_one(self):
        self.reset_files()
        self.clear_outputs()

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not path:
            self.status.configure(text="Tab 1: đã hủy tải ảnh.")
            return

        self.paths = [Path(path)]
        self.batch = False

        self.show_preview(self.paths[0])
        self.clear_outputs(keep_original=True)

        self.btn_detect.configure(state="normal")
        self.status.configure(text="Tab 1: phiên ảnh mới đã reset kết quả cũ. Bấm Detect.")

    def load_batch(self):
        self.reset_files()
        self.clear_outputs()

        paths = filedialog.askopenfilenames(
            filetypes=[("Images/Archive", "*.jpg *.jpeg *.png *.bmp *.webp *.zip *.rar")]
        )

        if not paths:
            self.status.configure(text="Tab 1: đã hủy tải ảnh hàng loạt.")
            return

        self.paths = self.collect_files([Path(p) for p in paths])
        self.batch = len(self.paths) >= 2

        if self.paths:
            self.show_preview(self.paths[0])
            self.clear_outputs(keep_original=True)

        self.set_pdf_visible(self.batch)
        self.btn_detect.configure(state="normal")
        self.btn_pdf.configure(state="disabled")

        self.status.configure(
            text=f"Tab 1: phiên batch mới đã reset kết quả cũ, tải {len(self.paths)} ảnh."
        )

    def collect_files(self, paths):
        out = []

        for path in paths:
            suffix = path.suffix.lower()

            if suffix in IMAGE_EXTS:
                out.append(path)
            elif suffix in ARCHIVE_EXTS:
                out += self.extract_archive(path)

        return out

    def extract_archive(self, path):
        td = tempfile.TemporaryDirectory()
        self.tempdirs.append(td)

        folder = Path(td.name)

        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(folder)
        else:
            try:
                import rarfile

                with rarfile.RarFile(path, "r") as r:
                    r.extractall(folder)
            except Exception:
                messagebox.showwarning(
                    "RAR",
                    "RAR cần cài rarfile và công cụ giải nén RAR trên máy.",
                )
                return []

        return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    def show_preview(self, path):
        img = imread_unicode(path)

        if img is not None:
            self.original_box.show(img, 1400, 540)

    def detect(self):
        if not self.paths:
            return

        self.busy = True
        self.status.configure(text="Tab 1: đang detect và OCR ảnh...")
        self.update()

        try:
            self.model_service.reset_tracker()
            model = self.model_service.load()

            self.samples = []

            for path in self.paths:
                sample = self.process_image(model, path)

                if sample:
                    self.samples.append(sample)

            if not self.samples:
                self.status.configure(
                    text="Tab 1: không đọc được ảnh hoặc không phát hiện biển số."
                )
                return

            self.show_sample(self.samples[0])

            if self.batch:
                self.btn_pdf.configure(state="normal")
                self.status.configure(
                    text=f"Tab 1: hoàn tất detect/OCR {len(self.samples)} ảnh. Có thể xuất PDF."
                )
            else:
                self.status.configure(text="Tab 1: hoàn tất detect/OCR ảnh đơn.")

        finally:
            self.busy = False

    def process_image(self, model, path):
        img = imread_unicode(path)

        if img is None:
            return None

        dets = detect_predict(model, img, CONF, IMGSZ, DEVICE, HALF)
        plates = []

        for idx, det in enumerate(dets, 1):
            raw_crop = crop_with_pad(img, det["box"])
            crop = enhance_crop(raw_crop)

            q = crop_quality(
                crop,
                det["conf"],
                det["box"],
                img.shape[1],
                img.shape[0],
            )

            text, best_crop, quality = self.ocr_service.best_text(
                [dict(crop=crop, quality=q)]
            )

            plates.append(
                dict(
                    idx=idx,
                    crop=best_crop,
                    text=text,
                    box=det["box"],
                    conf=det["conf"],
                    quality=quality,
                )
            )

        boxed = draw_boxes(img, dets, len(dets), 0)

        return dict(
            name=path.name,
            original=img,
            boxed=boxed,
            plates=plates,
        )

    def show_sample(self, sample):
        self.original_box.show(sample["original"], 1400, 540)
        self.bbox_box.show(sample["boxed"], 1400, 540)

        crop_items = []
        ocr_records = []

        for plate in sample["plates"]:
            crop_items.append(
                {
                    "crop": plate["crop"],
                    "quality": plate.get("quality", 0),
                }
            )

            ocr_records.append(
                {
                    "crop": plate["crop"],
                    "text": plate.get("text") or "OCR_UNCLEAR",
                    "quality": plate.get("quality", 0),
                }
            )

        self.side.show_crops(crop_items)
        self.side.show_ocr(ocr_records)

    def export_report(self):
        if not self.samples:
            return

        out = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
        )

        if not out:
            return

        export_pdf(self.samples, out)
        messagebox.showinfo("PDF", f"Đã xuất báo cáo:\n{out}")
