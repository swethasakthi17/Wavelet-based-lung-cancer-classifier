DTCWT WAVEFORM COEFFICIENTS EXTRATION:
import numpy as np
import pywt
from scipy.fftpack import dct
import cv2
import matplotlib.pyplot as plt

def dtcwt_feature_extraction(image):
    # Denoise the image using total variation denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Perform DTCWT on the denoised image
    coeffs = pywt.dwt2(denoised_image, 'bior3.3', mode='symmetric')

    # Extract features from each level
    features = []

    cA, (cH, cV, cD) = coeffs

    # Feature extraction from approximation coefficients (cA)
    features.extend(compute_features(cA))
    
    # Visualize approximation coefficients (cA)
    plt.subplot(2, 2, 1)
    plt.imshow(cA, cmap='gray')
    plt.title('Approximation Coefficients')

    # Feature extraction from horizontal, vertical, and diagonal detail coefficients (cH, cV, cD)
    features.extend(compute_features(cH))
    features.extend(compute_features(cV))
    features.extend(compute_features(cD))

    # Visualize detail coefficients
    plt.subplot(2, 2, 2)
    plt.imshow(cH, cmap='gray')
    plt.title('Horizontal Detail Coefficients')

    plt.subplot(2, 2, 3)
    plt.imshow(cV, cmap='gray')
    plt.title('Vertical Detail Coefficients')

    plt.subplot(2, 2, 4)
    plt.imshow(cD, cmap='gray')
    plt.title('Diagonal Detail Coefficients')

    plt.show()

    return features

def compute_features(coefficients):
    # Example: compute mean, variance, skewness, and kurtosis
    mean = np.mean(coefficients)
    variance = np.var(coefficients)
    skewness = np.mean(((coefficients - mean) / np.std(coefficients)) ** 3)
    kurtosis = np.mean(((coefficients - mean) / np.std(coefficients)) ** 4) - 3
    # Add more features as needed
    return [mean, variance, skewness, kurtosis]

# Example usage:
image_path = 'C:/Users/shrid/Downloads/archive/Data/test/adenocarcinoma/000122.png'
loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

# Check if the image is loaded successfully
if loaded_image is not None:
    features = dtcwt_feature_extraction(loaded_image)
    print(features)
else:
    print(f"Failed to load the image from the path: {image_path}")
import numpy as np
import pywt
from scipy.fftpack import dct
import cv2
import matplotlib.pyplot as plt

def dtcwt_feature_extraction(image):
    # Denoise the image using total variation denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Perform DTCWT on the denoised image
    coeffs = pywt.dwt2(denoised_image, 'bior3.3', mode='symmetric')

    # Extract features from each level
    features = []

    cA, (cH, cV, cD) = coeffs

    # Feature extraction from approximation coefficients (cA)
    features.extend(compute_features(cA))
    
    # Visualize approximation coefficients (cA)
    plt.subplot(2, 2, 1)
    plt.imshow(cA, cmap='gray')
    plt.title('Approximation Coefficients')

    # Feature extraction from horizontal, vertical, and diagonal detail coefficients (cH, cV, cD)
    features.extend(compute_features(cH))
    features.extend(compute_features(cV))
    features.extend(compute_features(cD))

    # Visualize detail coefficients
    plt.subplot(2, 2, 2)
    plt.imshow(cH, cmap='gray')
    plt.title('Horizontal Detail Coefficients')

    plt.subplot(2, 2, 3)
    plt.imshow(cV, cmap='gray')
    plt.title('Vertical Detail Coefficients')

    plt.subplot(2, 2, 4)
    plt.imshow(cD, cmap='gray')
    plt.title('Diagonal Detail Coefficients')

    plt.show()

    return features

def compute_features(coefficients):
    # Example: compute mean, variance, skewness, and kurtosis
    mean = np.mean(coefficients)
    variance = np.var(coefficients)
    skewness = np.mean(((coefficients - mean) / np.std(coefficients)) ** 3)
    kurtosis = np.mean(((coefficients - mean) / np.std(coefficients)) ** 4) - 3
    # Add more features as needed
    return [mean, variance, skewness, kurtosis]

# Example usage:
image_path = 'C:/Users/shrid/Downloads/archive/Data/test/adenocarcinoma/000122.png'
loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

# Check if the image is loaded successfully
if loaded_image is not None:
    features = dtcwt_feature_extraction(loaded_image)
    print(features)
else:
    print(f"Failed to load the image from the path: {image_path}")
import numpy as np
import pywt
from scipy.fftpack import dct
import cv2
import matplotlib.pyplot as plt

def dtcwt_feature_extraction(image):
    # Denoise the image using total variation denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Perform DTCWT on the denoised image
    coeffs = pywt.dwt2(denoised_image, 'bior3.3', mode='symmetric')

    # Extract features from each level
    features = []

    cA, (cH, cV, cD) = coeffs

    # Feature extraction from approximation coefficients (cA)
    features.extend(compute_features(cA))

    # Feature extraction from horizontal, vertical, and diagonal detail coefficients (cH, cV, cD)
    features.extend(compute_features(cH))
    features.extend(compute_features(cV))
    features.extend(compute_features(cD))

    return features

def compute_features(coefficients):
    # Example: compute mean, variance, skewness, and kurtosis
    mean = np.mean(coefficients)
    variance = np.var(coefficients)
    skewness = np.mean(((coefficients - mean) / np.std(coefficients)) ** 3)
    kurtosis = np.mean(((coefficients - mean) / np.std(coefficients)) ** 4) - 3
    # Add more features as needed
    return [mean, variance, skewness, kurtosis]

def visualize_features(selected_features):
    # Reshape the selected features into a smaller array
    selected_mask = np.reshape(selected_features, (4, 4))

    # Display the selected features mask
    plt.imshow(selected_mask, cmap='gray')
    plt.title('Selected Features')
    plt.show()
