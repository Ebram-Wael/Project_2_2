import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")
image_size = (28, 28)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("L") 
    img = img.resize(image_size)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

img_path = "../Data/dataset_split/predict/sample_image2.png"


img_array = preprocess_image(img_path)
preds = model.predict(img_array)
predicted_class = np.argmax(preds)

print(f"Predicted class: {predicted_class}")
