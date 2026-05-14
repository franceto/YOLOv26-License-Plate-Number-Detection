import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from lp_app.ui import theme as T
from lp_app.services.vision import fit_frame


def ctk_image_bgr(img, max_w, max_h):
    img = fit_frame(img, max_w, max_h)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


def photo_bgr_fill_height(img, max_w, max_h):
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb)), 1, 1

    scale = max_h / max(h, 1)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return ImageTk.PhotoImage(Image.fromarray(rgb)), new_w, new_h


class ImageBox(ctk.CTkFrame):
    def __init__(self, parent, title, h=360):
        super().__init__(
            parent,
            fg_color=T.WHITE,
            border_width=1,
            border_color=T.BORDER,
        )

        self.img = None
        self.canvas_h = h

        ctk.CTkLabel(
            self,
            text=title,
            font=T.FONT_BOLD,
            text_color=T.TEXT,
            fg_color=T.WHITE,
        ).pack(anchor="w", padx=10, pady=(8, 4))

        holder = tk.Frame(self, bg=T.WHITE)
        holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(
            holder,
            bg=T.WHITE,
            height=h,
            highlightthickness=0,
            xscrollincrement=20,
        )
        self.canvas.pack(side="top", fill="both", expand=True)

        self.xbar = tk.Scrollbar(
            holder,
            orient="horizontal",
            command=self.canvas.xview,
        )
        self.xbar.pack(side="bottom", fill="x")

        self.canvas.configure(xscrollcommand=self.xbar.set)
        self.clear("Chưa có ảnh")

    def show(self, bgr, max_w=1500, max_h=360):
        target_h = max(260, min(int(max_h), 900))

        self.canvas_h = target_h
        self.canvas.configure(height=target_h)
        self.canvas.delete("all")

        self.img, iw, ih = photo_bgr_fill_height(bgr, max_w, target_h)

        visible_w = max(self.canvas.winfo_width(), max_w)
        x = max(0, (visible_w - iw) // 2)
        y = max(0, (target_h - ih) // 2)

        self.canvas.create_image(x, y, image=self.img, anchor="nw")
        self.canvas.configure(
            scrollregion=(0, 0, max(iw, visible_w), target_h)
        )

    def clear(self, text="Chưa có ảnh"):
        self.canvas.delete("all")
        self.img = None
        self.canvas.configure(bg=T.WHITE)

        self.canvas.create_text(
            20,
            self.canvas_h // 2,
            text=text,
            anchor="w",
            fill=T.MUTED,
            font=("Segoe UI", 11),
        )

        self.canvas.configure(scrollregion=(0, 0, 1, self.canvas_h))


class ResultSidePanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(
            parent,
            fg_color=T.PANEL,
            border_width=1,
            border_color=T.BORDER,
            width=260,
        )

        self.grid_propagate(False)

        self.crop_imgs = []
        self.ocr_imgs = []

        ctk.CTkLabel(
            self,
            text="Kết quả",
            font=T.TITLE,
            text_color=T.TEXT,
        ).pack(anchor="w", padx=14, pady=(14, 8))

        self.tabs = ctk.CTkTabview(
            self,
            fg_color=T.PANEL,
            segmented_button_fg_color=T.WHITE,
            segmented_button_selected_color="#E5E7EB",
            segmented_button_selected_hover_color="#DADDE3",
            segmented_button_unselected_color=T.WHITE,
            segmented_button_unselected_hover_color="#F0F0F0",
            text_color=T.BLACK,
        )
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.crop_tab = self.tabs.add("Ảnh crop realtime")
        self.ocr_tab = self.tabs.add("OCR")

        self.crop_frame = ctk.CTkScrollableFrame(
            self.crop_tab,
            fg_color=T.SOFT,
        )
        self.crop_frame.pack(fill="both", expand=True)

        self.ocr_frame = ctk.CTkScrollableFrame(
            self.ocr_tab,
            fg_color=T.SOFT,
        )
        self.ocr_frame.pack(fill="both", expand=True)

        self.clear()

    def clear(self):
        self.show_crops([])
        self.show_ocr([])

    def show_crops(self, items):
        for widget in self.crop_frame.winfo_children():
            widget.destroy()

        self.crop_imgs.clear()

        if not items:
            ctk.CTkLabel(
                self.crop_frame,
                text="Chưa có ảnh crop.",
                text_color=T.MUTED,
            ).pack(anchor="w", padx=8, pady=8)
            return

        for item in items:
            card = ctk.CTkFrame(
                self.crop_frame,
                fg_color=T.WHITE,
                border_width=1,
                border_color=T.BORDER,
            )
            card.pack(fill="x", padx=8, pady=8)

            img = ctk_image_bgr(item["crop"], 220, 115)
            self.crop_imgs.append(img)

            label = tk.Label(card, image=img, bg=T.WHITE)
            label.image = img
            label.pack(padx=8, pady=8)

    def show_ocr(self, records):
        for widget in self.ocr_frame.winfo_children():
            widget.destroy()

        self.ocr_imgs.clear()

        if not records:
            ctk.CTkLabel(
                self.ocr_frame,
                text="Chưa có kết quả OCR.",
                text_color=T.MUTED,
            ).pack(anchor="w", padx=8, pady=8)
            return

        for record in records:
            card = ctk.CTkFrame(
                self.ocr_frame,
                fg_color=T.WHITE,
                border_width=1,
                border_color=T.BORDER,
            )
            card.pack(fill="x", padx=8, pady=8)

            img = ctk_image_bgr(record["crop"], 220, 105)
            self.ocr_imgs.append(img)

            label = tk.Label(card, image=img, bg=T.WHITE)
            label.image = img
            label.pack(padx=8, pady=(8, 4))

            ctk.CTkLabel(
                card,
                text=record.get("text", "OCR_UNCLEAR"),
                font=("Segoe UI", 15, "bold"),
                text_color=T.BLACK,
            ).pack(padx=8, pady=(0, 10))
