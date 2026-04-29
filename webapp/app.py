import os
import sys

from flask import Flask, render_template, request, url_for
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from webapp.services.inference import predict_with_heatmap, save_result_images  # noqa: E402


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No se envió ningún archivo (campo 'image').")

    f = request.files["image"]
    if not f or not f.filename:
        return render_template("index.html", error="Selecciona una imagen para subir.")

    try:
        pil = Image.open(f.stream)
    except Exception:
        return render_template("index.html", error="No pude leer el archivo como imagen.")

    try:
        result = predict_with_heatmap(pil)
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e))

    names = save_result_images(result, OUTPUT_DIR)
    original_url = url_for("static", filename=f"outputs/{names['orig_name']}")
    heatmap_url = url_for("static", filename=f"outputs/{names['overlay_name']}")

    return render_template(
        "index.html",
        result=result,
        original_url=original_url,
        heatmap_url=heatmap_url,
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
