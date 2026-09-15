# 🌱 Plant Disease Classification using Deep Learning

A deep learning project for classifying plant diseases from leaf images using **Transfer Learning with MobileNetV2**.

The model is trained using the **New Plant Diseases Dataset (Augmented)** and implemented with **TensorFlow/Keras** in Google Colab.

## 📌 Project Overview

Plant diseases can significantly affect crop production. Identifying diseases early can help farmers take appropriate action.

This project uses a Convolutional Neural Network (CNN) based approach with **MobileNetV2**, a pretrained model originally trained on ImageNet. Transfer learning is used to adapt the pretrained network for plant disease classification.

The project includes:

* Dataset loading and preprocessing
* Image normalization
* Data augmentation
* MobileNetV2 transfer learning
* Initial training with frozen pretrained layers
* Fine-tuning of the top 30 layers
* Early stopping
* Model checkpointing
* Training accuracy and loss visualization
* Saving the trained model
* Saving class names for later prediction

## 🧠 Model Architecture

The project uses **MobileNetV2** as the pretrained base model.

### Architecture

```text
Input Image (224 × 224 × 3)
          ↓
   Data Augmentation
          ↓
      MobileNetV2
   (Pretrained Model)
          ↓
Global Average Pooling
          ↓
     Dense Layer
     (128 neurons)
          ↓
       Dropout
       (30%)
          ↓
   Softmax Output
          ↓
 Plant Disease Class
```

## 📂 Dataset

The project uses the:

**New Plant Diseases Dataset (Augmented)**

The dataset contains images of plant leaves belonging to multiple disease and healthy classes.

The notebook expects the dataset to contain the following folders:

```text
New Plant Diseases Dataset(Augmented)/
├── train/
└── valid/
```

The training and validation directories are loaded using TensorFlow's `image_dataset_from_directory()`.

## ⚙️ Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Matplotlib
* Google Colab
* Google Drive

## 🔧 Configuration

The model uses the following configuration:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
```

Images are resized to:

```text
224 × 224 pixels
```

Pixel values are normalized to the range:

```text
0 – 1
```

using TensorFlow's `Rescaling(1./255)` layer.

## 🔄 Data Augmentation

To improve model generalization and reduce overfitting, the project applies:

* Random horizontal flipping
* Random rotation
* Random zoom

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

## 🚀 Training Process

Training is performed in two phases.

### Phase 1 — Transfer Learning

The pretrained MobileNetV2 layers are initially frozen:

```python
base_model.trainable = False
```

A new classification head is added on top of MobileNetV2.

The model uses:

```text
Optimizer: Adam
Learning Rate: 0.001
Loss: Categorical Crossentropy
Metric: Accuracy
```

### Phase 2 — Fine-Tuning

After the initial training, MobileNetV2 is partially unfrozen.

Only the last 30 layers are allowed to train:

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False
```

A smaller learning rate is used:

```python
Adam(learning_rate=1e-5)
```

This allows the pretrained model to adapt more carefully to the plant disease dataset.

## 🛑 Callbacks

The project uses two callbacks:

### Early Stopping

Training can stop when validation performance stops improving.

```python
EarlyStopping(
    patience=3,
    restore_best_weights=True
)
```

### Model Checkpoint

The best model is saved during training:

```python
ModelCheckpoint(
    "models/best_model.h5",
    save_best_only=True
)
```

## 💾 Model Output

The final trained model is saved as:

```text
models/plant_disease_model.h5
```

The class names are also saved in:

```text
class_names.txt
```

These files can be used later to build a prediction application.

## 📊 Training Results

The project generates training graphs showing:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

The results are saved as:

```text
training_results.png
```

Add your training-results screenshot here:

```text
![Training Results](training_results.png)
```

## 📁 Recommended Repository Structure

```text
plant-disease-classification/
│
├── deep_learning_model_training.ipynb
├── class_names.txt
├── training_results.png
│
├── models/
│   └── plant_disease_model.h5
│
├── README.md
│
└── requirements.txt
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the notebook

Open:

```text
deep_learning_model_training.ipynb
```

You can run the notebook using **Google Colab** or Jupyter Notebook.

### 3. Prepare the dataset

Download the New Plant Diseases Dataset (Augmented) and place it in your Google Drive.

The notebook currently expects the dataset at:

```text
/content/drive/MyDrive/archive (1).zip
```

### 4. Run the notebook

Run the cells in order:

```text
Mount Google Drive
       ↓
Load Dataset
       ↓
Preprocess Images
       ↓
Data Augmentation
       ↓
Build MobileNetV2 Model
       ↓
Train Model
       ↓
Fine-Tune Model
       ↓
Save Model
       ↓
Plot Results
```

## 📦 Requirements

Example `requirements.txt`:

```text
tensorflow
matplotlib
numpy
```

## 🎯 Future Improvements

Possible improvements for this project include:

* Build a **Gradio web interface**
* Add real-time image prediction
* Deploy the model online
* Add confidence scores
* Add a confusion matrix
* Add precision, recall, and F1-score
* Convert the model to TensorFlow Lite
* Improve the model using additional fine-tuning
* Create a mobile application for plant disease detection

## 👨‍💻 Author

**Idrees Khan**

Artificial Intelligence Student

## ⭐ Project Goal

The main goal of this project is to demonstrate how **deep learning and transfer learning can be used for automated plant disease classification**.

This project is also part of my journey toward building practical **AI and Machine Learning projects** and sharing them on GitHub.

---

⭐ If you find this project useful, consider giving the repository a star!

