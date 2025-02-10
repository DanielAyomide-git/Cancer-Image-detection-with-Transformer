from flask import Flask, request, jsonify
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import io

app = Flask(__name__)

# Load the processor and model for ViT
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Define a route for the home page (index.html)
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Define a route to handle image uploads and processing
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream)

    # Preprocess the image and run it through the ViT model
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():  # We only need inference here
        outputs = model(**inputs)

    # Extract last hidden states
    last_hidden_states = outputs.last_hidden_state

    # For simplicity, we'll just return a dummy prediction. 
    # You can replace this with actual logic for cancer detection.
    # You might want to use a classifier head or perform more complex operations.
    prediction = "Cancer Detected" if torch.mean(last_hidden_states) > 0 else "No Cancer Detected"

    return jsonify({'success': True, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
