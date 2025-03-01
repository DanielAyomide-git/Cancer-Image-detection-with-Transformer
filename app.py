from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import io
import os

app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")


# Route to serve the HTML file
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/detect", methods=["POST"])
def detect():
    # Check if an image is provided
    if "file" not in request.files:
        return jsonify({"success": False, "result": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "result": "No selected file"})

    # Open the image and process it
    try:
        img = Image.open(file.stream)
        inputs = processor(images=img, return_tensors="pt")

        # Model inference (for feature extraction, adjust as per actual model usage)
        outputs = model(**inputs)

        # For demonstration purposes, returning dummy result (change logic as needed)
        result = "This is a demonstration only. Real cancer detection would require a specialized model."

        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify(
            {"success": False, "result": f"Error processing image: {str(e)}"}
        )


if __name__ == "__main__":
    app.run(debug=True)
