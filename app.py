from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
from efficientnet_pytorch import EfficientNet

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load('svm_model.pkl')

# Load EfficientNet for feature extraction
efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

def preprocess_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def extract_features(image):
    # Extract features using EfficientNet
    features = efficientnet.extract_features(image)
    return features.detach().numpy()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Save the uploaded image
        image_file = request.files['file']
        image_path = 'uploads/' + image_file.filename
        image_file.save(image_path)
        
        # Preprocess and extract features
        image = preprocess_image(image_path)
        features = extract_features(image)
        
        # Predict the disease
        prediction = model.predict(features.reshape(1, -1))
        
        # Display the result
        return render_template('result.html', prediction=prediction[0])
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
