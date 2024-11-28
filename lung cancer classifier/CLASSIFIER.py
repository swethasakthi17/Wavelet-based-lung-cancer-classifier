CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore', message="DataFrame is highly fragmented")
import pickle
base_dir = "C:/Users/shrid/Downloads/archive/Data/train"
class_folder_names = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 'normal', 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa',]
class_folder_names
class_dict = {'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib':"Adenocarcinoma",'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa':'large.cell.carcinoma','normal':'normal','squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa':"squamous.cell.carcinoma"}
class_dict
import os

base_dir = 'C:/Users/shrid/Downloads/archive/Data/'
class_folder_names = ['train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib']

image_paths = []

for cls in class_folder_names:
    class_dir = os.path.join(base_dir, cls)
    print(class_dir)  # Print out the constructed directory path for debugging
    if not os.path.exists(class_dir):
        print(f"Directory {class_dir} does not exist.")
        continue
    
    print(cls)
    for file_name in os.listdir(class_dir):
        if file_name.split(".")[-1] in ['jpg', 'png']:
            image_paths.append(os.path.join(class_dir, file_name))
            
print("Total images = ", len(image_paths))
print("----------------------")

print(image_paths[0:10])  # Print out the first 10 image paths for inspection
import os

base_dir = 'C:/Users/shrid/Downloads/archive/Data/'
class_folder_names = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

image_paths = []

for cls in class_folder_names:
    class_dir = os.path.join(base_dir, 'train', cls)  # Assuming images are in the 'train' subdirectory
    print(class_dir)  # Print out the constructed directory path for debugging
    if not os.path.exists(class_dir):
        print(f"Directory {class_dir} does not exist.")
        continue
    
    print(cls)
    for file_name in os.listdir(class_dir):
        if file_name.split(".")[-1] in ['jpg', 'png']:
            image_paths.append(os.path.join(class_dir, file_name))
            
print("Total images = ", len(image_paths))
print("----------------------")

print(image_paths[0:10])  # Print out the first 10 image paths for inspection
classes = []

# Ensure image_paths contains correct paths
print(image_paths)

for image_path in image_paths:
    classes.append(image_path.split('\\')[-2])

print(classes[0:5])
Assuming you want to resize all images to (width, height)
target_size = (224, 224)

inputs = []

for i in tqdm(image_paths):
    image = load_img(i, target_size=target_size)
    img_array = img_to_array(image)
    inputs.append(img_array)

X = np.array(inputs)
le = LabelEncoder()
y = le.fit_transform(classes)
y = np.array(y)

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print("Train data = ", X_train.shape,  y_train.shape)
print("Test data = ", X_test.shape,  y_test.shape)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

print("Flattened Train data = ", X_train_flattened.shape,  y_train.shape)
print("Flattened Test data = ", X_test_flattened.shape,  y_test.shape)
# Create a dictionary to store indices of samples for each class
unique_classes = np.unique(y_train)
class_indices = {class_id: np.where(y_train == class_id)[0] for class_id in unique_classes}
# ploting total samples for each class

images_count = [len(class_indices[key]) for key in class_indices.keys()]

fig = px.bar(x=class_dict.values(), y=images_count, color= class_dict.values())

fig.update_layout(xaxis_title='Disease', yaxis_title='Count', title="Total samples for each class", )
fig.update_traces(texttemplate='%{y}', textposition='inside')

fig.show()
# Displaying sample images for each class randomly.

plt.figure(figsize=(12, 5))

for i, (class_id, indices) in enumerate(class_indices.items()):
    random_index = np.random.choice(indices)
    random_image = X_train[random_index].astype(np.uint8)

    plt.subplot(1, len(unique_classes), i + 1)
    plt.imshow(random_image)
    plt.title(class_dict[class_folder_names[class_id]])
    plt.axis('on')

plt.show()
# Displaying 7 sample images for each class randomly.

selected_classes = [0,1,2, 3]
total_images_per_class = 5

plt.figure(figsize=(15, 5))

for c, selected_class in enumerate(selected_classes):
    
    # Get indices of samples for the selected class
    indices_for_selected_class = np.where(y_train == selected_class)[0]
    random_indices = np.random.choice(indices_for_selected_class, total_images_per_class, replace=False)

    # Display images for the current selected class
    for i, idx in enumerate(random_indices):
        plt.subplot(len(selected_classes), total_images_per_class, c * total_images_per_class + i + 1)
        plt.imshow(X_train[idx].astype(np.uint8))
        plt.title(class_dict[class_folder_names[selected_class]])
        plt.axis('off')

plt.tight_layout()
plt.show()
# Reshape the image data to a 2D array
num_samples, height, width, channels = X_train.shape

X_train_reshaped = X_train.reshape(num_samples, height * width * channels)
X_train_flattened.shape
# Apply PCA to reduce dimensions to 2 for visualization

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_flattened)
X_train_pca.shape
# Visualize the PCA-transformed data

hover_text = [f"Index: {index}" for index in range(len(X_train_pca))]

fig = px.scatter(x=X_train_pca[:, 0], y=X_train_pca[:, 1], color=y_train, hover_name=hover_text, symbol=y_train, title='PCA Visualization of Image Classes')
fig.update_traces(marker=dict(size=15))
fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')
fig.update_layout(coloraxis_showscale=False)

fig.show()
# Finding outliers by talking high distance sum values

pca_sums = np.sum(X_train_pca, axis=1)

outlier_indexes = []
for idx, row in enumerate(pca_sums):
    if row>15000:
        print(row, idx)
        outlier_indexes.append(idx)
# Displaying outlier images

plt.figure(figsize=(15, 4))

for i, outlier in enumerate(outlier_indexes):
    outlier_image = X_train[outlier] 

    plt.subplot(1, len(outlier_indexes), i+1)
    plt.imshow(outlier_image)
    title = class_folder_names[y_train[outlier]]
    plt.title(title)
    plt.axis('off')

plt.show()
pca_df = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2'])
pca_df['Class'] = y_train
pca_df['Index'] = pca_df.index
pca_df.head()
# Create side-by-side box plots for PC1 and PC2 using Plotly

fig = px.box(pca_df, x='Class', y=['PC1', 'PC2'], points="all", facet_col="variable",
             title='Box Plots of PCA - Principal Components 1 and 2 by Class', hover_data={'Index': True})
fig.update_layout(width=1200, height=500)
fig.show()
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),

}
accuracies = {}

for name, model in models.items():
    
    model.fit(X_train_flattened, y_train)        
    predictions = model.predict(X_test_flattened)
    
    accuracy = accuracy_score(y_test, predictions)
    accuracies[name] = accuracy

    print(f"{name} accuracy: {accuracy}")
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

print("Best model = ", best_model )
# Train the best model
best_model.fit(X_train_flattened, y_train)
best_predictions = best_model.predict(X_test_flattened)

best_predictions
# Get weights

if best_model_name == 'Random Forest' or best_model_name == 'Decision Tree':
    coefficients = best_model.feature_importances_
    print(coefficients.shape)

elif best_model_name == 'Logistic Regression':
    coefficients = best_model.coef_.ravel()
    print(coefficients.shape)
else:
    coefficients = None
# ploting weights distribution

if coefficients is not None:
    fig = px.histogram(x=coefficients, nbins=50, labels={'x': 'Coefficient Value'}, title='Distribution of Coefficients (Weights)')
    fig.update_layout(bargap=0.1)
    fig.update_traces(opacity=0.7)
    fig.show()
report = classification_report(y_test, best_predictions)
print(report)
# Calculate and display the confusion matrix for the best model
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title(f'Confusion Matrix for the Best Model based on Accuracy ({best_model_name})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
dataset = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\shrid\OneDrive\Documents\MATLAB\archive\Data\test",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True
)
image_size = 256
batch_size = 32
channels=3
epochs=50
class_names = dataset.class_names
len(dataset)
plt.figure(figsize=(10,10))
for i, j in dataset.take(1):
    for k in range(12):
        ax = plt.subplot(3,4,k+1)
        plt.imshow(i[k].numpy().astype('int64'))
        plt.title(class_names[j[k]])
        plt.axis('off')
print(j.numpy().shape)
def get_ds(ds,train_split=0.8, val_split=0.1,test_split=0.1, shuffle=True, shuffle_size = 10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed =12)
    train_size = int(train_split*ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds,test_ds
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds =  val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

