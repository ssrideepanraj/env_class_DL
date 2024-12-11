# Environmental Classification Using Deep Learning

This repository contains the implementation of a deep learning-based project for classifying environmental images into distinct categories, such as `Forest`, `Residential`, etc. The project utilizes a Convolutional Neural Network (CNN) with additional feature extraction using fractal dimensions to enhance classification accuracy. The model is trained and tested using the EuroSAT dataset, with integration of fractal features to improve performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Model Architecture](#model-architecture)
- [Performance Evaluation](#performance-evaluation)
- [Setup and Usage](#setup-and-usage)
- [Deployment](#deployment)
- [Contributors](#contributors)

---

## Project Overview
The primary objective of this project is to classify environmental regions using RGB satellite imagery. Fractal dimensions are integrated as additional features to enhance model performance. The project includes:
- Data preprocessing and augmentation.
- Feature extraction using fractal dimensions.
- Model building and training using ResNet50.
- Evaluation of model performance with metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
- Deployment-ready trained model.

## Technologies Used
- **Programming Languages**: Python
- **Libraries**: TensorFlow, NumPy, Pandas, Matplotlib, OpenCV, PIL, sklearn
- **Deep Learning Architecture**: ResNet50

## Dataset
The project uses the **EuroSAT RGB Dataset** containing satellite images in `.jpg` format. The dataset includes 10 classes, such as `Forest` and `Residential`. It is split into 80% for training and 20% for testing, with an additional 20% of the training data used for validation.

## Feature Extraction
Fractal dimensions are calculated for each image to capture texture-based features:
- Box-counting algorithm is used for fractal dimension computation.
- These features are combined with the image data for training.

## Model Architecture
The classification model is built using:
- ResNet50 as the base architecture for feature extraction.
- Additional fractal features combined with image data.
- A final dense layer for classification into 10 categories.

### Training Configuration
- **Batch Size**: 32
- **Image Size**: 64x64
- **Epochs**: 100 (with EarlyStopping)
- **Optimizer**: Adam
- **Callbacks**: EarlyStopping and ModelCheckpoint

## Performance Evaluation
The model is evaluated using:
- **Accuracy**: Overall correctness of predictions.
- **Precision, Recall, F1-Score**: Class-wise performance.
- **Confusion Matrix**: Detailed class-level insights.

## Setup and Usage
### Prerequisites
- Python 3.8+
- TensorFlow 2.11
- GPU-compatible CUDA drivers

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repository-name>.git
   cd <repository-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```
5. Deploy the model (instructions in `deployment/` folder).

## Deployment
The trained model is saved as `best_model.keras` and is ready for deployment using TensorFlow Serving or a web framework like Flask. Refer to the `deployment/` folder for instructions.

## Contributors
- **Srideepanraj S.** - Model Development, Feature Engineering, and Deployment.
- **Batch Member** - Web Application Deployment and Documentation.

---

Feel free to raise an issue or contribute to improve the project!
"# env_class_DL" 
"# env_class_DL" 
"# env_class_DL" 
"# env_class_DL" 
