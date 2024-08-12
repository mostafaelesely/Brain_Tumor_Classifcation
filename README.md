# Brain Tumor MRI Classification

This project focuses on classifying brain tumor MRI images into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**. The model is built using the **EfficientNetB7** architecture, fine-tuned for the specific task. This repository contains the code to train, validate, test, and deploy the model.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Prediction](#prediction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview

The goal of this project is to develop a robust deep learning model that can accurately classify brain tumor MRI images into one of the following categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

We use the EfficientNetB7 model, a powerful and efficient architecture, to handle the image classification task.

## Dataset

The dataset is sourced from the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) on Kaggle. The dataset is divided into training and testing directories, each containing images organized in subfolders corresponding to the different classes.

- **Training Data Directory**: `/kaggle/input/brain-tumor-mri-dataset/Training`
- **Testing Data Directory**: `/kaggle/input/brain-tumor-mri-dataset/Testing`

## Model Architecture

The model is built on top of the pre-trained EfficientNetB7 model. The architecture is as follows:
- **Base Model**: EfficientNetB7 (pre-trained on ImageNet)
- **Global Average Pooling Layer**
- **Dropout Layer (0.5)**
- **Dense Layer**: 512 units with ReLU activation and L2 regularization
- **Dropout Layer (0.5)**
- **Output Layer**: 4 units with Softmax activation (for multi-class classification)

## Training and Evaluation

The model is trained using the following settings:
- **Optimizer**: Adamax with a learning rate of 0.001
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 16
- **Image Size**: 224x224
- **Data Augmentation**: Random rotation, shifting, shearing, zooming, and horizontal flipping

Training is monitored with Early Stopping and Model Checkpoint callbacks to prevent overfitting.

### Training History

The training and validation accuracy and loss are plotted to visualize the model's performance over time.

### Test Evaluation

The model achieved a **Test Accuracy** of **99.7%** on the testing dataset.

## Prediction

You can use the trained model to predict the class of a new MRI image. Example usage:

```python
image_path = '/path/to/image.jpg'
predicted_class = predict_image(image_path, model, img_size, class_labels)
print(f'Predicted class: {predicted_class}')
```

## Dependencies

The project uses the following libraries:
- TensorFlow
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

Ensure you have the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-mri-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd brain-tumor-mri-classification
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Use the model for prediction:
   ```bash
   python predict.py --image_path /path/to/image.jpg
   ```

## Results

The model performed exceptionally well on the test data, achieving high accuracy with minimal loss. The final test accuracy was **99.7%**, demonstrating the model's effectiveness in classifying brain tumor types.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.


