from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from PIL import Image
import cv2

def bgr_reader(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    bio = BytesIO()
    pil.save(bio, format="JPEG", quality=90)
    bio.seek(0)
    return ImageReader(bio)

def draw_img(c, img, x, y, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = iw * s, ih * s
    c.drawImage(bgr_reader(img), x + (w - nw) / 2, y + (h - nh) / 2, nw, nh)

def export_pdf(samples, out_path):
    c = canvas.Canvas(str(out_path), pagesize=A4)
    W, H = A4

    for i, s in enumerate(samples, 1):
        c.setFont("Helvetica-Bold", 15)
        c.drawString(30, H - 35, f"License Plate OCR Report - Image {i}")

        c.setFont("Helvetica", 10)
        c.drawString(30, H - 52, str(s.get("name", "")))

        c.setFont("Helvetica-Bold", 11)
        c.drawString(30, H - 75, "1. Original image")
        draw_img(c, s["original"], 30, H - 270, 240, 180)

        c.drawString(310, H - 75, "2. Detected image")
        draw_img(c, s["boxed"], 310, H - 270, 240, 180)

        y = H - 305
        c.drawString(30, y, "3. Crops and OCR results")
        y -= 20

        plates = s.get("plates", [])
        if not plates:
            c.setFont("Helvetica", 11)
            c.drawString(30, y, "No plate detected.")
        else:
            for j, p in enumerate(plates, 1):
                if y < 130:
                    c.showPage()
                    y = H - 60
                draw_img(c, p["crop"], 30, y - 75, 150, 65)
                c.setFont("Helvetica-Bold", 13)
                c.drawString(200, y - 25, f"OCR {j}: {p.get('text', 'OCR_UNCLEAR')}")
                y -= 95

        c.showPage()

    c.save()