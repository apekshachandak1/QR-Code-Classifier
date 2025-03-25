import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load dataset path
dataset_path = r"path/to/dataset/Assignment Data-20250323T075006Z-001/Assignment Data"
categories = ['First Print', 'Second Print']

# Function to load images
def load_images(dataset_path, categories):
    data, labels = [], []
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            print(f"Error: Directory not found - {category_path}")
            continue  # Skip if directory doesn't exist
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            img = cv2.resize(img, (128, 128))  # Resize to (128,128)
            data.append(img)
            labels.append(label)
    
    return np.array(data), np.array(labels)

# Load images
data, labels = load_images(dataset_path, categories)

# Ensure dataset is not empty
if len(data) == 0:
    raise RuntimeError("Dataset loading failed! Check file paths and dataset structure.")

# Normalize and reshape for CNN
data_cnn = data / 255.0
X_cnn = data_cnn.reshape(-1, 128, 128, 1)

# Train-Test Split
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, labels, test_size=0.2, random_state=42)

# Train CNN Model
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn))

# Evaluate CNN Model
cnn_preds = (cnn_model.predict(X_test_cnn) > 0.5).astype('int')

print("CNN Accuracy:", accuracy_score(y_test_cnn, cnn_preds))
print(classification_report(y_test_cnn, cnn_preds))

# Save Model
cnn_model.save('qr_classifier.h5')

print("Model saved as 'qr_classifier.h5'")
