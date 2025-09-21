import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# -------------------------------
# Load the trained model
# -------------------------------
model = load_model('model/cats_dogs_model.h5')

# -------------------------------
# Function to predict the image
# -------------------------------
def predict_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Load and preprocess image
    img = image.load_img(file_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    if pred[0][0] > 0.5:
        result_text.set("Prediction: Dog")
    else:
        result_text.set("Prediction: Cat")

    # Show selected image in GUI
    img_display = Image.open(file_path)
    img_display = img_display.resize((200, 200))
    img_display = ImageTk.PhotoImage(img_display)
    image_label.config(image=img_display)
    image_label.image = img_display

# -------------------------------
# Create GUI window
# -------------------------------
root = tk.Tk()
root.title("Cat vs Dog Classifier")
root.geometry("300x400")

# Button to upload image
upload_btn = Button(root, text="Upload Image", command=predict_image)
upload_btn.pack(pady=10)

# Label to display prediction
result_text = tk.StringVar()
result_label = Label(root, textvariable=result_text, font=("Arial", 16))
result_label.pack(pady=10)

# Label to show image
image_label = Label(root)
image_label.pack(pady=10)

# Start GUI
root.mainloop()
