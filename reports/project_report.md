# Smart Plant Disease Detection System Using CNN and Transfer Learning

## 1. Introduction

Plant diseases are a major threat to agriculture and food security. Early detection of plant diseases helps farmers take timely action and reduce crop loss. Traditional manual inspection requires expert knowledge and takes time. Therefore, an automated plant disease detection system can be very useful.

## 2. Problem Statement

The main problem is to classify plant leaf images into healthy or diseased classes using deep learning techniques.

## 3. Objectives

- To collect plant leaf image datasets.
- To preprocess leaf images.
- To train a CNN or transfer learning model.
- To classify plant diseases accurately.
- To build a simple web interface for prediction.

## 4. Existing System

In the existing system, plant disease detection is mostly done manually by farmers or agricultural experts. This process can be slow, costly, and sometimes inaccurate.

## 5. Proposed System

The proposed system uses deep learning to automatically classify plant leaf diseases. The user uploads a leaf image, and the system predicts the disease class with a confidence score.

## 6. Methodology

1. Dataset collection
2. Image preprocessing
3. Data augmentation
4. Model training using CNN / MobileNetV2
5. Model evaluation
6. Prediction through Streamlit web app

## 7. Algorithms Used

- Convolutional Neural Network
- Transfer Learning using MobileNetV2

## 8. Requirements

### Software Requirements

- Python 3.10+
- TensorFlow
- Keras
- OpenCV
- Streamlit
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow

### Hardware Requirements

Minimum:
- Intel i3/i5 processor
- 8 GB RAM
- 5 GB free storage

Recommended:
- Intel i5/i7 processor
- 16 GB RAM
- NVIDIA GPU

## 9. Expected Output

The system predicts plant disease class and confidence score.

Example:

Disease: Black Rot  
Confidence: 94.62%  
Recommendation: Remove infected leaves and apply suitable fungicide.

## 10. Future Scope

- Add more plant species
- Mobile app development
- Real-time camera detection
- Regional language support
- Cloud deployment
