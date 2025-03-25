# QR Code Classifier

## Introduction
This project implements a QR code classification model using Convolutional Neural Networks (CNN). The model differentiates between 'First Print' (authentic) and 'Second Print' (counterfeit) QR code images using deep learning techniques. 

## Approach and Methodology

### 1. Dataset & Preprocessing
- **Dataset:** Images labeled as ‘First Print’ and ‘Second Print’.
- **Preprocessing:**
  - Converted to grayscale for reduced complexity.
  - Resized to **128×128 pixels** for uniformity.
  - Normalized pixel values to **[0,1]**.
  - Applied data augmentation for robustness.

### 2. Model Architecture
- **Feature Extraction:** CNN layers with ReLU activation.
- **Downsampling:** Max-pooling layers.
- **Regularization:** Dropout layers to prevent overfitting.
- **Output Layer:** Sigmoid activation for binary classification.

### 3. Training Process
- **Data Split:** 80% Training | 20% Validation.
- **Loss Function:** Binary Cross-Entropy.
- **Optimizer:** Adam.
- **Regularization Techniques:** Batch Normalization & Dropout.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

## Experiments and Results

### 1. Performance Metrics
- **Accuracy:** 95%
- **Precision:** 100%
- **Recall:** 89.47%
- **F1 Score:** 94.44%

### 2. Confusion Matrix
| Actual \ Predicted | 0 (Negative) | 1 (Positive) |
|------------------|-------------|-------------|
| **0 (Negative)** | 21 (TN)     | 0 (FP)      |
| **1 (Positive)** | 2 (FN)      | 17 (TP)     |

### 3. Observations
- The model achieved **95% accuracy**.
- Precision of **100%** means no false positives.
- Recall of **89.5%** suggests 2 false negatives.
- The **F1-score of 94.4%** balances precision and recall.

## How It Works
1. Users upload an image (drag & drop, browse, or manual path entry).
2. The script loads the trained model and processes the image.
3. The model predicts the category (‘First Print’ or ‘Second Print’).
4. The prediction is displayed along with the original image.

## Deployment Considerations
- **User Interface:** Web or mobile-based for easy image uploads.
- **Cloud Deployment:** Use cloud services for scalability.
- **Latency Optimization:** TensorFlow Lite for real-time processing.
- **Security Measures:** Adversarial attack prevention, robust feature extraction.

## Future Enhancements
- **Improve Model Accuracy:** Train with a larger dataset and fine-tune hyperparameters.
- **Optimize Model for Deployment:** Use lightweight models for real-time inference.
- **Enhance Robustness:** Implement adaptive preprocessing for varying scanning conditions.
- **Real-Time Classification:** Develop a web-based real-time classifier.
- **QR Code Detection:** Add verification logic before classification.

## Conclusion
This project successfully developed a deep learning-based QR code classification model, achieving **95% accuracy**. Future work will focus on optimizing deployment, improving robustness, and enhancing real-time classification capabilities.

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/apekshachandak1/QR-Code-Classifier.git
cd QR-Code-Classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Model
```bash
python run.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
