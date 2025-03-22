from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rembg import remove
from PIL import Image, ImageOps
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

def process_image(image_bytes):
    # Remove background
    output_image = remove(image_bytes)

    # Convert bytes to PIL image
    image = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # Convert to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGRA)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGRA2GRAY)

    # Load OpenCV face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]  # Get the first detected face

        # Expand cropping area
        padding_x = int(w * 0.2)  # 20% extra width
        padding_y = int(h * 0.4)  # 40% extra height

        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(open_cv_image.shape[1], w + 2 * padding_x)
        h = min(open_cv_image.shape[0], h + 2 * padding_y)

        # Crop the image around the face
        cropped_image = open_cv_image[y:y+h, x:x+w]
    else:
        # If no face is detected, return the original image
        cropped_image = open_cv_image

    # Convert back to PIL Image
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2RGBA))

    # Resize to 350x450 pixels (3.5 cm × 4.5 cm at 100 DPI)
    final_image = cropped_pil.resize((350, 450), Image.LANCZOS)

    # Create a white background
    white_background = Image.new("RGBA", final_image.size, (255, 255, 255, 255))

    # Paste the cropped image onto the white background
    white_background.paste(final_image, (0, 0), final_image)

    # Convert back to RGB (removing alpha channel)
    final_image_rgb = white_background.convert("RGB")

    # Save the final image to BytesIO
    output_io = io.BytesIO()
    final_image_rgb.save(output_io, format="PNG")
    output_io.seek(0)

    return output_io

@app.route('/', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return jsonify({"error": "Invalid image file"}), 400

    # Process the image
    output_io = process_image(uploaded_file.read())

    return send_file(output_io, mimetype='image/png')
