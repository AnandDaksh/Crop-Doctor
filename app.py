import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Correct paths to your model files
model_paths = [
    r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model.h5',
    r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model2.h5',
    r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model3.h5',
    r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model4_v2.h5',
    r'C:\Users\user7\Desktop\Crop Doctor\crop_doctor-main\my_model5.h5',
]

# Verify if the paths exist
for path in model_paths:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file or directory found at {path}")

# Debugging: Print TensorFlow/Keras version
print(f"TensorFlow Version: {tf.__version__}")

# Load your Keras models
models = []
for path in model_paths:
    try:
        model = tf.keras.models.load_model(path)
        models.append(model)
        print(f"Loaded model from {path}")
    except Exception as e:
        print(f"Error loading model from {path}: {e}")

# Labels for the predictions
crop_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', "Alstonia Scholaris diseased",
    "Alstonia Scholaris healthy", "Arjun diseased", "Arjun healthy", "Citrus_Black spot", "Citrus_canker", "Citrus_greening",
    "Citrus_healthy", "Guava diseased", "Guava healthy", "Jamun diseased", "Jamun healthy", "Pomegranate diseased", "Pomegranate healthy",
    "Pongamia Pinnata diseased", "Pongamia Pinnata healthy", "Bael diseased", "Basil healthy", "Jatropa diseased", "Jatropa healthy",
    "Lemon diseased", "Lemon healthy", "Mango diseased", "Mango healthy", "Rose_Healthy_Leaf", "Rose_Rust", "Rose_sawfly_Rose_slug",
    "Soybean_healthy", "Sugarcane_Banded_Chlorosis", "Sugarcane_BrownRust", "Sugarcane_Brown_Spot", "Sugarcane_Grassy shoot",
    "Sugarcane_Pokkah Boeng", "Sugarcane_Sett Rot", "Sugarcane_Viral Disease", "Sugarcane_Yellow Leaf", "Tea_algal_spot", "Tea_brown_blight",
    "Tea_gray_blight", "Tea_healthy", "Tea_helopeltis", "Tea_red_spot", "Blueberry_healthy", "Cherry_healthy", "Cherry_Powdery_mildew",
    "Chinar diseased", "Chinar healthy", "Tulsi_bacterial", "Tusli_fungal", "Tulsi_healthy"
]

def getResult(image_path, model, labels):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x.astype('float32') / 255.0  # Normalize the image data to 0-1

    # Predict the class
    predictions = model.predict(x)
    class_id = np.argmax(predictions)  # Get the index of the max logit/probability
    return labels[class_id]  # Fetch the label using index from the list

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)
        
        # Choose a model randomly
        chosen_model = random.choice(models)
        
        # Predict using the chosen model
        result = getResult(file_path, chosen_model, crop_labels)
        return f"Prediction: {result}"
        
    return None

if __name__ == '__main__':
    app.run(debug=True)
