import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("models/mask_detection_model.keras")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    prediction = model.predict(img_input)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    label = "with mask " if class_idx == 1 else "without mask "

    plt.imshow(img_rgb)
    plt.title(f'{label} ({confidence:.2f}%)')
    plt.axis('off')
    plt.show()


predict_image("4.jpg")
