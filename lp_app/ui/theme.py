import customtkinter as ctk

BG = "#F5F6F8"
PANEL = "#FFFFFF"
SOFT = "#FAFAFA"
BORDER = "#D9DEE7"
TEXT = "#111111"
MUTED = "#666666"
BLACK = "#000000"
RED = "#FF0000"
WHITE = "#FFFFFF"

FONT = ("Segoe UI", 13)
FONT_BOLD = ("Segoe UI", 13, "bold")
TITLE = ("Segoe UI", 20, "bold")

def setup_theme():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

def button_style():
    return dict(
        fg_color=WHITE,
        hover_color="#ECECEC",
        text_color=BLACK,
        border_width=1,
        border_color=BORDER,
        font=FONT_BOLD,
        height=34,
        corner_radius=4,
    )

def active_button(btn):
    btn.configure(fg_color=BLACK, text_color=RED)

def normal_button(btn):
    btn.configure(fg_color=WHITE, text_color=BLACK)