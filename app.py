from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os
import base64
import uuid

app = Flask(__name__)

# Load the trained model
model = load_model("blood_cell.h5")

# Define class labels
class_labels = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def predict_image_class(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_preprocessed = preprocess_input(img_resized.reshape((1, 128, 128, 3)))
    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))
    predicted_class_label = class_labels[predicted_class_idx]
    return predicted_class_label, confidence, img_rgb

# Home + Upload Route
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save file with a unique filename
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            file_path = os.path.join("static", filename)
            file.save(file_path)

            predicted_class_label, confidence, img_rgb = predict_image_class(file_path, model)

            # Convert image to base64
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')

            return render_template("result.html",
                                   class_label=predicted_class_label,
                                   confidence=f"{confidence:.2%}",
                                   img_data=img_str,
                                   image_name=filename)

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
