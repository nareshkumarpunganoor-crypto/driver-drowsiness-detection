# 🚗 Driver Drowsiness Detection – Alerts Sleepy Drivers

A **Deep Learning-based Driver Drowsiness Detection System** that identifies whether a driver is **DROWSY** or **NATURAL/ALERT** from facial images and webcam frames.

The project uses **MobileNetV2 Transfer Learning** for image classification and combines it with **OpenCV Haar Cascade face and eye detection** to provide a real-time drowsiness warning. When drowsiness is detected continuously, an **audio alarm** is triggered to alert the driver.

---

## 📌 Project Overview

Driver fatigue and drowsiness are important causes of road accidents. This project provides a computer-vision-based solution that continuously monitors the driver's facial condition.

The system:

* 📷 Captures the driver's face using a webcam
* 👤 Detects the driver's face using OpenCV
* 👁️ Detects the eyes using Haar Cascade
* 🧠 Classifies the facial state using MobileNetV2
* 😴 Identifies **DROWSY** and **NATURAL** states
* ⏱️ Uses a drowsiness counter to reduce instant false alarms
* 🔊 Activates an audio alarm when drowsiness persists
* 📊 Evaluates the trained model using accuracy, precision, recall and confusion matrix

---

## 🎯 Objectives

1. Develop a deep learning model for driver drowsiness detection.
2. Classify driver facial states into **DROWSY** and **NATURAL**.
3. Use transfer learning to improve classification performance.
4. Implement real-time webcam-based detection.
5. Detect prolonged eye closure using computer vision.
6. Generate an audio warning when drowsiness is detected.
7. Build a foundation for future real-world driver safety applications.

---

## 🧠 Technology Used

| Technology     | Purpose                   |
| -------------- | ------------------------- |
| Python         | Programming               |
| TensorFlow     | Deep Learning             |
| Keras          | Model development         |
| MobileNetV2    | Transfer-learning model   |
| OpenCV         | Computer Vision           |
| Haar Cascade   | Face and eye detection    |
| NumPy          | Numerical processing      |
| Matplotlib     | Visualization             |
| Scikit-learn   | Model evaluation          |
| Google Colab   | Model training            |
| Kaggle Dataset | Training and testing data |

---

## 📂 Dataset

The project uses the Kaggle **Drowsy Detection Dataset**.

Dataset:

**Drowsy Detection Dataset – Kaggle**

Classes:

```text
DROWSY
NATURAL
```

### Dataset Distribution

| Dataset   |    DROWSY |   NATURAL |     Total |
| --------- | --------: | --------: | --------: |
| Training  |     2,809 |     3,050 |     5,859 |
| Testing   |       757 |       726 |     1,483 |
| **Total** | **3,566** | **3,776** | **7,342** |

The training data is further divided into training and validation subsets during model development.

---

## 🏗️ Project Architecture

```text
                 ┌─────────────────────┐
                 │   Webcam / Image    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Face Detection    │
                 │   Haar Cascade      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Face Preprocessing  │
                 │ Resize: 128 × 128   │
                 │ Normalization       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    MobileNetV2      │
                 │ Transfer Learning   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ DROWSY / NATURAL    │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Eye Detection          Drowsiness Counter
          Haar Cascade                 │
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │   Drowsiness Alert  │
                 │     🔊 ALARM        │
                 └─────────────────────┘
```

---

## 🧠 Model – MobileNetV2

The project uses **MobileNetV2** with ImageNet pretrained weights as the feature extraction backbone.

The pretrained convolutional layers are initially frozen and additional classification layers are added for the two-class drowsiness problem.

### Model Structure

```text
Input Image
     │
     ▼
128 × 128 × 3
     │
     ▼
Rescaling
     │
     ▼
MobileNetV2
     │
     ▼
Global Average Pooling
     │
     ▼
Dropout
     │
     ▼
Dense Layer – 128 neurons
     │
     ▼
Dropout
     │
     ▼
Sigmoid Output
     │
     ├── 0 → DROWSY
     │
     └── 1 → NATURAL
```

---

## ⚙️ Image Preprocessing

Input images are resized to:

```text
128 × 128 pixels
```

Pixel values are normalized before being passed to the model.

Data augmentation is applied during training using:

* Rotation
* Width shifting
* Height shifting
* Zoom
* Horizontal flipping

These techniques help the model learn from variations in facial position and appearance.

---

## 👁️ Eye Detection

In addition to the deep learning classifier, the system uses OpenCV Haar Cascade eye detection.

The eye detector monitors the upper region of the detected face.

If eyes are not detected continuously, the **closed-eye counter** increases.

```text
Eyes detected
     │
     ├── YES → Reset closed-eye counter
     │
     └── NO  → Increase closed-eye counter
```

This additional signal is combined with the MobileNetV2 prediction to improve real-time drowsiness detection.

---

## 🚨 Drowsiness Alert Logic

The system does not immediately trigger an alarm from a single frame.

Instead, it uses a counter.

```text
Drowsiness detected
       │
       ▼
Drowsy Counter
       │
       ├── Below threshold → Continue monitoring
       │
       └── Threshold reached
                    │
                    ▼
             🚨 DROWSINESS ALERT
                    │
                    ▼
               🔊 Alarm
```

This helps reduce unnecessary alerts caused by short-term predictions or normal blinking.

---

## 📊 Model Performance

During model evaluation, the MobileNetV2 model achieved approximately:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **94.00%** |
| Precision | **93.81%** |
| Recall    | **93.94%** |

### Confusion Matrix

```text
                 Predicted
               DROWSY  NATURAL

Actual DROWSY    712      45

Actual NATURAL    44     682
```

The model was evaluated on the held-out test dataset.

> **Note:** Test-set performance does not by itself guarantee equivalent performance in real-world driving conditions. Lighting, camera position, facial appearance, glasses, occlusion, and different drivers can affect detection performance.

---

## 💻 Project Workflow

### Step 1 – Dataset Preparation

The dataset is downloaded and organized into:

```text
dataset/
└── Drowsy_datset/
    ├── train/
    │   ├── DROWSY/
    │   └── NATURAL/
    │
    └── test/
        ├── DROWSY/
        └── NATURAL/
```

### Step 2 – Data Preprocessing

Images are resized to:

```text
128 × 128
```

and normalized before training.

### Step 3 – Model Training

MobileNetV2 pretrained on ImageNet is used for transfer learning.

### Step 4 – Model Evaluation

The trained model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

### Step 5 – Webcam Detection

The webcam captures frames continuously.

### Step 6 – Drowsiness Detection

Each frame is analyzed using:

* Face detection
* Eye detection
* MobileNetV2 classification
* Drowsiness counter

### Step 7 – Alert

If drowsiness persists:

```text
🚨 DROWSINESS ALERT!
🔊 WAKE UP!
```

The audio alarm is activated.

---

## 📁 Project Structure

```text
driver-drowsiness-detection/
│
├── README.md
│
├── notebooks/
│   └── driver_drowsiness_detection.ipynb
│
├── model/
│   └── drowsiness_mobilenetv2.keras
│
├── dataset/
│   └── Drowsy_datset/
│       ├── train/
│       │   ├── DROWSY/
│       │   └── NATURAL/
│       │
│       └── test/
│           ├── DROWSY/
│           └── NATURAL/
│
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── drowsiness_alarm.wav
│
└── requirements.txt
```

> Adjust the folder names above to match the files actually committed to the repository.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/nareshkumarpunganoor-crypto/driver-drowsiness-detection.git
```

Move into the project directory:

```bash
cd driver-drowsiness-detection
```

Install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn scipy
```

For Google Colab, the required deep learning libraries can be installed/imported directly in the notebook.

---

## ▶️ Running the Project

### Train the Model

Open the project notebook in Google Colab and run the dataset preparation and training cells.

### Test an Image

Provide a driver image to the prediction section.

The model returns:

```text
DROWSY
```

or

```text
NATURAL
```

along with the prediction confidence.

### Run Webcam Detection

Start the webcam detection section and allow browser camera permission.

The system then continuously monitors the driver.

---

## 🔊 Alert System

The project includes an audio alert mechanism.

When the drowsiness counter reaches the configured threshold:

```text
🚨 DROWSINESS ALERT!
       ↓
    WAKE UP!
       ↓
   🔊 ALARM
```

The alarm continues while the drowsiness condition remains active.

---

## 📈 Results and Visualizations

The project can generate visualizations such as:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Confusion matrix
* Classification report
* Sample predictions

These visualizations help evaluate the performance of the trained model.

---

## 🌟 Key Features

* 🧠 Deep Learning-based classification
* 🚗 Driver drowsiness monitoring
* 📷 Webcam support
* 👤 Face detection
* 👁️ Eye detection
* ⚡ MobileNetV2 transfer learning
* 📊 Model evaluation
* ⏱️ Drowsiness counter
* 🔊 Audio warning
* 💻 Google Colab compatible
* 📈 Performance visualization

---

## 🔮 Future Enhancements

The current project can be extended with:

* 👁️ Eye Aspect Ratio (EAR)
* 😮 Yawning detection
* 🧑‍🦰 Facial landmark detection
* 🧠 Head-pose estimation
* 📱 Mobile application
* 🌐 Web-based deployment
* 🚘 In-vehicle deployment
* 📍 GPS location tracking
* 📩 Emergency notifications
* 🗄️ Drowsiness event database
* 🤖 Real-time edge-device inference
* 🔔 Multiple levels of driver alerts

A multimodal system combining eye closure, yawning, head pose, and temporal behavior could provide a more comprehensive driver-monitoring solution.

---

## ⚠️ Limitations

The system is a **deep-learning micro-project and prototype**, not a certified automotive safety system.

Performance can be affected by:

* Poor lighting
* Camera quality
* Face angle
* Sunglasses
* Face occlusion
* Different driver appearances
* Camera placement
* Background conditions
* Short-term facial movements

The reported test accuracy is based on the available dataset and should not be interpreted as guaranteed real-world driving accuracy.

---

## 🎓 Project Information

**Project Title:**

### Driver Drowsiness Detection – Alerts Sleepy Drivers

**Project Type:**
Deep Learning Micro Project

**Domain:**
Artificial Intelligence / Computer Vision / Deep Learning

**Model:**
MobileNetV2 Transfer Learning

**Framework:**
TensorFlow / Keras

**Computer Vision:**
OpenCV

**Development Platform:**
Google Colab

---

## 👨‍💻 Author

### Punganoor Naresh Kumar

B.Tech – Artificial Intelligence & Data Science

GitHub:
https://github.com/nareshkumarpunganoor-crypto

---

## 📌 Repository

**Driver Drowsiness Detection – Deep Learning Micro Project**

[https://github.com/nareshkumarpunganoor-crypto/driver-drowsiness-detection](https://colab.research.google.com/drive/1YNMvocqFkHFLxQBjwFLvt79gHDZIYSjg?usp=sharing)

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ and following the project for future updates.

---

### 📜 Disclaimer

This project is developed for **educational and research purposes**. It should not be considered a replacement for certified driver-monitoring or automotive safety systems.
