from lp_app.ui.theme import setup_theme
from lp_app.ui.app import App

if __name__ == "__main__":
    setup_theme()
    app = App()
    app.mainloop()