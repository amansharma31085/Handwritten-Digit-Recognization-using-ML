# 🤖 Handwritten-Digit-Recognization-using-ML
This project is a **Handwritten Digit Recognition Web Application** built using **PyTorch** and **Flask**. It leverages a **Convolutional Neural Network (CNN)** to accurately predict handwritten digits drawn by users through a web interface. The application also provides **interpretability features**, showing the model's confidence in its predictions and a visual representation of how the model processes the input.

-----

## ✨ Features

  * **Handwritten Digit Recognition**: Accurately classifies handwritten digits (0-9) drawn by the user on a canvas.
  * **Real-time Prediction**: Provides instant predictions as soon as the user finishes drawing.
  * **Probability Visualization**: Displays a bar chart showing the probability distribution for each digit, indicating the model's confidence.
  * **Model Interpretability**: Visualizes the feature maps from the first convolutional layer of the CNN, offering insights into how the model "sees" the input image.
  * **Responsive Web Interface**: Built with Flask and HTML/CSS/JavaScript for an interactive user experience.

-----

## 💻 Technology Stack

  * **Python**: Primary programming language.
  * **PyTorch**: Deep learning framework for building and training the CNN model.
  * **Flask**: Web framework for serving the application.
  * **Gunicorn**: WSGI HTTP Server for Python web applications.
  * **Numpy**: For numerical operations, especially image processing.
  * **Pillow (PIL)**: For image manipulation.
  * **Matplotlib**: Used for generating probability and interpretability visualizations.
  * **HTML, CSS, JavaScript (jQuery)**: For the web front-end.

-----

## 🧠 Model Architecture

The core of this project is a custom **Convolutional Neural Network (CNN)**, `MnistModel`, defined in `train.py` and `torch_serve/model.py`. This model is designed for image classification tasks, specifically for handwritten digits.

The architecture consists of:

  * **Two Convolutional Blocks**: Each block includes two `Conv2d` layers followed by `ReLU` activation, a `MaxPool2d` layer for down-sampling, and a `Dropout` layer for regularization.
      * **Block 1**: Converts (N, 1, 28, 28) input to (N, 32, 10, 10) output.
      * **Block 2**: Converts (N, 32, 10, 10) input to (N, 128, 3, 3) output.
  * **Fully Connected Layers**:
      * A `Linear` layer (dense3) that transforms the flattened output of the convolutional layers (128\*3\*3) to 32 features, followed by `ReLU` and `Dropout`.
      * A final `Linear` layer (dense4) that maps the 32 features to `classes` (10 for digits 0-9), followed by a `log_softmax` activation for probability distribution.

-----

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed. It's recommended to use a virtual environment.
You can install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

### Running the Application

1.  **Train the Model (Optional)**: If you want to retrain the model or ensure you have the `mnist.pt` checkpoint, run `train.py`. The trained model will be saved in the `checkpoint` directory.

    ```bash
    python train.py
    ```

2.  **Start the Flask Web App**: Run the `app.py` file. This will start the Flask development server.

    ```bash
    python app.py
    ```

    The application will typically be accessible at `http://127.0.0.1:5000` as defined in `config.py`.

### Usage

Once the application is running:

1.  Open your web browser and navigate to `http://127.0.0.1:5000`.
2.  **Draw a digit** in the provided canvas using your mouse.
3.  Click the **"predict"** button to get the model's prediction, probability distribution, and interpretability image.
4.  Click the **"clear"** button to reset the canvas and make a new prediction.

-----

## 🧪 Testing

Basic tests for the Flask application are available in `test_app.py`, which verify the status codes for the home page and non-existent routes.

```bash
python test_app.py
```

-----

## ⚙️ TorchServe Integration

The `torch_serve` directory contains files (`model.py`, `server.py`, `index_to_name.json`) prepared for deploying the model with **TorchServe**, allowing for scalable model serving. The `server.py` file demonstrates how to save the model as a TorchScript for deployment and includes a commented-out command for `torch-model-archiver`.
