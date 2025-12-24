# Gender Detection and Age Prediction

A deep learning project designed to automatically estimate the age and detect the gender of a person from facial images. This system utilizes a Multi-Output Convolutional Neural Network (CNN) trained on the UTKFace dataset.

# 📝 Project Overview
The "Gender Detection and Age Prediction" system leverages computer vision and deep learning techniques to analyze facial features. The model takes an image as input and simultaneously predicts:

Gender: Binary classification (Male/Female).

Age: Continuous regression output (Approximate age in years).

This project demonstrates the application of multi-task learning where a single neural network architecture shares features to solve two distinct problems concurrently.

# 📂 Dataset
The project uses the UTKFace dataset, a large-scale face dataset with a long age span (ranging from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity.

- Input Shape: Images are resized to 128x128 pixels and converted to grayscale.

- Normalization: Pixel values are normalized to a range of 0-1.

# 🛠 Technologies Used
- Language: Python 3

- Deep Learning Framework: TensorFlow / Keras 

- Data Manipulation: Pandas, NumPy 

- Visualization: Matplotlib, Seaborn 

- Environment: Jupyter Notebook / Kaggle Kernel

# 🧠 Model Architecture
The model is built using the Keras Functional API to handle multiple outputs. It consists of a shared convolutional base followed by separate branches for age and gender predictions.

**Shared Layers (Feature Extraction)**

1) Conv2D: 32 filters, 3x3 kernel, ReLU activation

2) MaxPooling2D: 2x2 pool size

3) Conv2D: 64 filters, 3x3 kernel, ReLU activation

4) MaxPooling2D: 2x2 pool size

5) Conv2D: 128 filters, 3x3 kernel, ReLU activation

6) MaxPooling2D: 2x2 pool size

7) Conv2D: 256 filters, 3x3 kernel, ReLU activation

8) MaxPooling2D: 2x2 pool size

9) Flatten

**Output Branches**
The network splits after flattening to perform two tasks:

**Gender Branch:**

- Dense (256, ReLU) + Dropout (0.3)

- Output Layer: Dense (1 unit, sigmoid activation) for binary classification.

**Age Branch:**

- Dense (256, ReLU) + Dropout (0.3)

- Output Layer: Dense (1 unit, relu activation) for regression.

**Compilation**

- Optimizer: Adam

- Loss Functions:

Gender: binary_crossentropy

Age: mean_absolute_error (MAE) 

# 🚀 Getting Started

**Prerequisites**

Ensure you have the following libraries installed:
```bash
pip install tensorflow pandas numpy matplotlib seaborn tqdm
```

**Usage**
1) Clone the repository.

2) Download the UTKFace dataset and update the BASE_DIR path in the notebook.

3) Run the GenderDetectionAgePrediction.ipynb notebook to preprocess data, train the model, and view results.

# 📊 Results
The model outputs training and validation accuracy/loss graphs for both tasks.

Gender Accuracy: High accuracy achieved using binary cross-entropy loss.

- Validation Accuracy: 89.20 %

Age Prediction: Minimized Mean Absolute Error (MAE) for age estimation.

- MAE: 6.78 Years

# Example Output:

Original Gender: Female, Original Age: 26, Predicted Gender: Female, Predicted Age: 27
