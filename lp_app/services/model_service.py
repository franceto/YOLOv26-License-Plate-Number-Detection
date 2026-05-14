from ultralytics import YOLO
from lp_app.config import MODEL_PATH

class ModelService:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None

    def load(self):
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Không tìm thấy model: {self.model_path}")
            self.model = YOLO(str(self.model_path))
        return self.model

    def reset_tracker(self):
        if self.model is not None:
            try:
                self.model.predictor = None
            except Exception:
                pass