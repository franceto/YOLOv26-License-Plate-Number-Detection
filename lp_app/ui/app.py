import customtkinter as ctk

from lp_app.config import APP_TITLE
from lp_app.ui import theme as T
from lp_app.ui.image_tab import ImageTab
from lp_app.ui.realtime_tab import RealtimeTab
from lp_app.services.model_service import ModelService
from lp_app.services.ocr_service import OCRService


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.configure(fg_color=T.BG)
        self.current_tab = "Tab 1 - Ảnh"
        self._restoring_tab = False

        self.fullscreen()

        self.model_service = ModelService()
        self.ocr_service = OCRService()

        self.tabs = ctk.CTkTabview(
            self,
            fg_color=T.BG,
            segmented_button_fg_color=T.WHITE,
            segmented_button_selected_color="#E5E7EB",
            segmented_button_selected_hover_color="#DADDE3",
            segmented_button_unselected_color=T.WHITE,
            segmented_button_unselected_hover_color="#F0F0F0",
            text_color=T.BLACK,
            command=self.on_tab_change,
        )
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)

        self.tab_image = self.tabs.add("Tab 1 - Ảnh")
        self.tab_video = self.tabs.add("Tab 2 - Video")
        self.tab_webcam = self.tabs.add("Tab 3 - Webcam")

        self.image_page = ImageTab(self.tab_image, self.model_service, self.ocr_service)
        self.video_page = RealtimeTab(self.tab_video, self.model_service, self.ocr_service, mode="video")
        self.webcam_page = RealtimeTab(self.tab_webcam, self.model_service, self.ocr_service, mode="webcam")

        self.image_page.pack(fill="both", expand=True)
        self.video_page.pack(fill="both", expand=True)
        self.webcam_page.pack(fill="both", expand=True)

        self.tabs.set("Tab 1 - Ảnh")
        self.after(200, self.bring_to_front)

    def fullscreen(self):
        try:
            self.state("zoomed")
        except Exception:
            self.attributes("-zoomed", True)

        self.minsize(1100, 720)

    def bring_to_front(self):
        try:
            self.lift()
            self.focus_force()
            self.state("zoomed")
        except Exception:
            pass

    def get_page(self, tab_name):
        if tab_name == "Tab 1 - Ảnh":
            return self.image_page
        if tab_name == "Tab 2 - Video":
            return self.video_page
        if tab_name == "Tab 3 - Webcam":
            return self.webcam_page
        return None

    def on_tab_change(self):
        if self._restoring_tab:
            return

        new_tab = self.tabs.get()
        old_page = self.get_page(self.current_tab)

        if old_page and hasattr(old_page, "is_busy") and old_page.is_busy():
            self._restoring_tab = True
            self.after(10, self.restore_old_tab)
            return

        self.current_tab = new_tab
        self.after_idle(self.update_idletasks)

    def restore_old_tab(self):
        self.tabs.set(self.current_tab)
        self._restoring_tab = False
