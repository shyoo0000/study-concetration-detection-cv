
# Real-time Attention Tracking with Computer Vision

This project is a Python-based application that uses a custom-trained machine learning model to analyze a user's attention level in real-time via a webcam. The model classifies the user's state into one of six categories related to focus and engagement. It is designed to be lightweight enough for integration with IoT platforms like Home Assistant for creating smart environmental automations.

## Features

- **6-Class Attention State Classification:** The model goes beyond simple "focused" vs. "not focused" states, providing more nuanced feedback by classifying into:
  - **Engaged:** `engaged`, `confused`, `frustrated`
  - **Not Engaged:** `looking_away`, `bored`, `drowsy`
- **Lightweight Model:** Built on **MobileNetV2** architecture, ensuring efficient performance on low-power hardware like Raspberry Pi or small servers.
- **Real-time Inference:** Capable of processing webcam streams to provide continuous feedback.
- **Ready for IoT Integration:** The final model is exported to `.tflite` format, making it easy to deploy with tools like DOODS on Home Assistant.

## How It Works

The system operates in two main stages: training and deployment.

### 1. Model Training

The model was trained using a supervised learning approach with TensorFlow and Keras.

- **Dataset:** A custom dataset was created by combining a public student-engagement dataset with personalized images for each of the six classes.
- **Transfer Learning:** The MobileNetV2 model, pre-trained on ImageNet, was used as a base. Only the final classification layer was retrained on the custom dataset.
- **Optimization Techniques:**
  - **Undersampling:** To address dataset imbalance, classes were down-sampled to match the count of the smallest class, preventing initial model bias.
  - **Class Weights:** To handle the varying difficulty of classifying features, custom weights were applied during training. For example, the `engaged` class was given a higher weight to encourage the model to learn its subtle features, while easily recognizable classes like `looking_away` were given lower weights.

    ```python
    # Example of class weight logic
    if name == 'looking_away':
        class_weights[index] = 0.3
    elif name == 'engaged':
        class_weights[index] = 2.0
    ...
    ```

### 2. Deployment & Inference

The trained model is designed to be used with a webcam stream for real-time analysis.

- **Face Detection:** An OpenCV Haar Cascade classifier first detects the location of a face in the video frame.
- **Image Preprocessing:** The detected face region is cropped, resized to 224x224 pixels, and normalized to match the model's input requirements.
- **Classification:** The preprocessed image is fed into the TensorFlow Lite model, which outputs a prediction for one of the six attention states.
- **Integration (Example):** The output can be sent to an MQTT broker or a Home Assistant API to trigger automations, such as adjusting lights or sending notifications.

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow
- OpenCV (`opencv-python`)
- NumPy

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/attention-tracker.git
    cd attention-tracker
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Download the Pre-trained Model:**
    Place the `concentration_model_final_weighted.tflite` file in the project's root directory. *(You can link to your model file in the GitHub Releases section)*.

2.  **Run the Real-time Demo:**
    Execute the demo script to start the webcam and see real-time classification.
    ```bash
    python demo.py
    ```

## Training Your Own Model

The training process was conducted in a Google Colab notebook. You can find the complete notebook (`training_notebook.ipynb`) in this repository. To train your own model, you will need to:

1.  Prepare your dataset with the following folder structure:
    ```
    dataset/
    ├── engaged/
    │   ├── img1.jpg
    │   └── ...
    ├── confused/
    │   └── ...
    └── (and so on for all 6 classes)
    ```
2.  Upload your dataset to Google Drive or the Colab environment.
3.  Open the notebook in Google Colab, update the file paths, and run the cells sequentially.

## Limitations

- The model's accuracy is highly dependent on lighting conditions and camera angles similar to those in the training dataset.
- It may misclassify subtle changes in posture as `drowsy` due to limitations in the training data.
- Performance can vary significantly between different individuals. For best results, supplement the dataset with your own images.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
