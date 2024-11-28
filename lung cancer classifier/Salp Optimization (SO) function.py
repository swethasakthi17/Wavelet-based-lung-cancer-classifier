Salp Optimization (SO) function
def so(feature_size, population_size=10, max_iter=100):
    # Initialize salp positions with continuous values between -1 and 1
    salps = np.random.rand(population_size, feature_size) * 2 - 1

    # Initialize best position and fitness value
    best_pos = salps[0].copy()
    best_fit = np.sum(salps[0])

    for t in range(1, max_iter + 1):
        # Update coefficient (a) to control step size
        a = 2 * (1 - (t / max_iter))

        for i in range(population_size):
            # Move salps towards the best position
            c1 = 2 * np.pi * np.random.rand(feature_size)  # Random coefficient
            c2 = 2 * np.pi * np.random.rand(feature_size)  # Random coefficient
            distance_to_best = np.abs(best_pos - salps[i])
            new_salp = salps[i] + (a * np.sin(c1) * distance_to_best) + (a * np.sin(c2))

            # Clip positions to stay within [-1, 1]
            new_salp = np.clip(new_salp, -1, 1)

            # Evaluate fitness
            fitness_val = np.sum(new_salp)

            # Update best position and fitness value
            if fitness_val > best_fit:
                best_fit = fitness_val
                best_pos = new_salp.copy()

    return best_pos

def combined_optimization(feature_size, population_size=10, max_iter=100):
    # Initialize salp positions with continuous values between -1 and 1
    salps = np.random.rand(population_size, feature_size) * 2 - 1

    # Initialize moth positions with continuous values between -1 and 1
    moths = np.random.rand(population_size, feature_size) * 2 - 1

    # Initialize best position and fitness value
    best_pos_so = salps[0].copy()
    best_fit_so = np.sum(salps[0])
    best_pos_mfo = moths[0].copy()
    best_fit_mfo = np.sum(moths[0])

    for t in range(1, max_iter + 1):
        # Update coefficients (a) and (b) to control step sizes for SO and MFO, respectively
        a = 2 * (1 - (t / max_iter))
        b = 1 - t / max_iter

        # Perform Salp Optimization
        for i in range(population_size):
            c1 = 2 * np.pi * np.random.rand(feature_size)
            c2 = 2 * np.pi * np.random.rand(feature_size)
            distance_to_best_so = np.abs(best_pos_so - salps[i])
            new_salp = salps[i] + (a * np.sin(c1) * distance_to_best_so) + (a * np.sin(c2))
            new_salp = np.clip(new_salp, -1, 1)
            fitness_val = np.sum(new_salp)
            if fitness_val > best_fit_so:
                best_fit_so = fitness_val
                best_pos_so = new_salp.copy()

        # Perform Moth Flame Optimization
        for i in range(population_size):
            distance_to_flame = np.linalg.norm(moths[i] - best_pos_mfo)
            new_moth = moths[i] * np.exp(-distance_to_flame ** 2) * b
            moths[i] += new_moth
            moths[i] = np.clip(moths[i], -1, 1)
            fitness_val = np.sum(moths[i])
            if fitness_val > best_fit_mfo:
                best_fit_mfo = fitness_val
                best_pos_mfo = moths[i].copy()

    return best_pos_so, best_pos_mfo

def random_classifier(num_classes, num_samples):
    # Generate random class labels for num_samples
    return np.random.randint(0, num_classes, num_samples)

# Example usage:
image_path = 'C:/Users/shrid/Downloads/archive/Data/test/adenocarcinoma/000122.png'
loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

# Check if the image is loaded successfully
if loaded_image is not None:
    features = dtcwt_feature_extraction(loaded_image)
    print("DTCWT Features:", features)
    
    # Now, apply combined optimization to select the best subset of features
    feature_size = len(features)
    selected_features_so, selected_features_mfo = combined_optimization(feature_size)
    print("Selected Features (SO):", selected_features_so)
    print("Selected Features (MFO):", selected_features_mfo)
    
    # Visualize the selected features
    visualize_features(selected_features_so)
    visualize_features(selected_features_mfo)

    # Generate random class labels using the random classifier
    num_classes = 5  # Number of classes
    num_samples = 10  # Number of samples
    class_labels = random_classifier(num_classes, num_samples)
    print("Random Class Labels:", class_labels)

else:
    print(f"Failed to load the image from the path: {image_path}")
import numpy as np 
import pandas as pd 
import os, random

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
