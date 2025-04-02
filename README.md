# HTFNet: Low-Light Image Enhancement Using Histogram and Spatial Features

## Overview
HTFNet is a deep learning model designed to enhance low-light images by combining histogram-based and spatial feature extraction. The model utilizes a combination of a histogram feature extractor, a convolutional spatial feature extractor, and a transformation function estimator to improve image brightness and contrast.

## Features
- **Histogram Feature Extraction**: Extracts color histograms from low-light images to capture intensity distributions.
- **Spatial Feature Extraction**: Uses convolutional layers to analyze structural and textural details.
- **Transformation Function Estimation**: Learns transformation parameters to enhance image quality.
- **Training with MSE Loss**: Uses Mean Squared Error (MSE) loss to optimize enhancement quality.
- **Supports CUDA**: Utilizes GPU acceleration if available for faster training and inference.

## Dataset
This implementation uses the **LOL (Low-Light) Dataset**, where each low-light image has a corresponding enhanced version. The dataset is stored in the following directory structure:

```
E:\Image Processing\project\LOLdataset\our485\low\       # Low-light images
E:\Image Processing\project\LOLdataset\our485\high\      # Enhanced images
```

## Installation
Ensure you have Python and the required libraries installed:

```sh
pip install torch torchvision pillow numpy matplotlib
```

## Usage

### 1. Training the Model
Run the following command to train the model:

```sh
python main.py
```

The model will be trained using images from the dataset and saved as `htfnet.pth`.

### 2. Enhancing an Image
Once trained, you can enhance a low-light image using:

```sh
python enhance.py --image_path "path/to/lowlight_image.jpg"
```

The enhanced image will be saved in the same directory as the original image with `_enhanced` appended to the filename.

## Model Components

### 1. Histogram Feature Extractor
Extracts and normalizes histograms for each image channel (RGB) and processes them through a fully connected layer.

### 2. Spatial Feature Extractor
A CNN-based feature extractor that captures texture and structure from the image.

### 3. Transformation Function Estimator
Combines histogram and spatial features to learn transformation parameters for image enhancement.

## Training Details
- **Optimizer**: Adam
- **Learning Rate**: 0.0002
- **Loss Function**: MSE Loss
- **Epochs**: 20
- **Batch Size**: 16

## Results
The model enhances low-light images by adjusting brightness and contrast while preserving texture details. Sample results can be visualized using `enhance_image()`.

## Contributing
If you'd like to contribute, please fork the repository and submit a pull request.

## License
This project is open-source and available under the MIT License.

