import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import logging

# ✅ Load trained model
model_path = r"C:\8 image cv project\internship project\QR CODE\qr_classifier.h5"

try:
    cnn_model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ✅ Function to preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not read {img_path}")
        return None
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)
    return img

# ✅ Function to display image and prediction
def show_image_with_prediction(img_path, prediction):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()

# ✅ User Input Options
print("\n📂 Choose Method to Upload Image :")
print("1️⃣ Drag & Drop → Simply drag the image into the terminal and press Enter.")
print("2️⃣ Browse File → A file selection window will open to pick an image.")
print("3️⃣ Manually Enter Path → Type the full image path in the terminal.")

choice = input("\n👉 Enter your choice (1/2/3): ").strip()
image_path = ""

# ✅ Step 1: Choose Input Method
if choice == "1":
    print("\n🔹 **Drag & Drop Mode:**")
    print("👉 Drag the image into this terminal and press Enter.")
    if len(sys.argv) < 2:
        image_path = input("📌 Enter image path manually: ").strip()
    else:
        image_path = sys.argv[1]

elif choice == "2":
    print("\n🔹 **Browse File Mode:**")
    print("👉 A file selection window will open. Select an image and click Open.")

    root = tk.Tk()
    root.attributes('-topmost', True)  # ✅ Force window to open on top
    root.withdraw()
    
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )

    root.destroy()  # ✅ Close the Tkinter instance after selection

elif choice == "3":
    print("\n🔹 **Manual Path Mode:**")
    print("👉 Type the full image path manually and press Enter.")
    image_path = input("📌 Enter image path manually: ").strip()

    # ✅ Improved check: Allow spaces, double quotes, and full paths
    if not os.path.exists(image_path):
        print("❌ Invalid file path! Please enter a correct manual path.")
        sys.exit(1)

else:
    print("❌ Invalid choice! Please enter the file path manually ! Please run the script again and enter a valid option.")
    sys.exit(1)

# ✅ Step 2: Validate Path
image_path = image_path.strip("& '\"")  # Remove extra characters (PowerShell issues)
if not os.path.exists(image_path):
    print(f"❌ Error: File not found - {image_path}")
    sys.exit(1)

print(f"\n✅ Selected file: {image_path}")

# ✅ Step 3: Preprocess & Predict
img = preprocess_image(image_path)
if img is not None:
    prediction = (cnn_model.predict(img) > 0.5).astype('int')
    categories = ['First Print', 'Second Print']
    result = categories[prediction[0][0]]
    
    # ✅ Show result in GUI
    show_image_with_prediction(image_path, result)
