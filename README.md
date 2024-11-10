
# Cataract Classification Project

This project is a binary image classification model designed to detect cataracts in eye images. It uses a deep learning model built on MobileNetV2, trained to classify images as either "cataract" or "normal." The trained model is deployed as an API using FastAPI to allow for easy integration and real-time inference.

## Project Structure
- **`model_training.py`**: Script for defining and training the model on the dataset.
- **`cataract_classifier_model.h5`**: The trained model file.
- **`utils.py`**: Utility functions for loading the model and preprocessing images.
- **`app.py`**: FastAPI application for deploying the model as an API.
- **Dataset**: Folder containing `train` and `test` data, each with subfolders for `cataract` and `normal` images.

## Requirements
- Python 3.8+
- Virtual environment (optional, but recommended)

## Setup Instructions

### 1. Environment Setup
1. Clone this repository to your local machine.
    ```bash
    git clone <repository-url>
    cd cataract-classification
    ```
2. Set up a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Model Training
The `model_training.py` script is used for model training and evaluation.

1. **Prepare the dataset**: Organize the images in the `train` and `test` folders with subfolders for `cataract` and `normal` images.
   
2. **Run the training script**:
    ```bash
    python model_training.py
    ```

   This will:
   - Load and preprocess the dataset.
   - Define and compile the model architecture (MobileNetV2-based).
   - Train the model for a set number of epochs.
   - Save the trained model as `cataract_classifier_model.h5`.

### 3.Model Evaluation

After training, the following metrics are evaluated on the test set:

Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
ROC Curve and AUC Score
These evaluations are included in the training script and plotted for easy interpretation of model performance.


### 4. API Deployment
The FastAPI application (`app.py`) provides an API endpoint for model inference.

1. **Run the FastAPI app**:
    ```bash
    uvicorn app:app --reload
    ```
   The server should start, and you will see output similar to:
   ```
   Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
   ```

2. **Testing the API**:
   - **Endpoint**: `POST /predict/`
   - **Request**: Upload an image file for prediction.
   - **Response**: Returns the predicted class (`cataract` or `normal`) and the confidence score.

#### Example Request
Using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@path_to_your_image.jpg"
```

### 4. API Documentation
The FastAPI server automatically generates API documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

To focus on API deployment and include an image showing the response, you can create a section in your `README.md` to explain the process of deploying the API and demonstrate how to use it with an example image showing the expected response.

Here’s a sample outline for your report, which you can include in the `README.md` file:

#### **Response:**

- **200 OK**:
  ```json
  {
    "class": "Cataract",
    "confidence": 0.95
  }
  ```

- **400 Bad Request** (if no image is uploaded):
  ```json
  {
    "detail": "No image uploaded"
  }
  ```


### 5.Challenges Faced

Class Imbalance: Worked on balancing the data and augmenting images for better results.
Overfitting: Addressed by using dropout layers and transfer learning to prevent overfitting.
Deployment: Ensured compatibility with FastAPI for smooth image-based inference.

### 6.Future Work

Experimenting with different architectures like ResNet or EfficientNet.
Implementing a web-based frontend for easier access to the API.