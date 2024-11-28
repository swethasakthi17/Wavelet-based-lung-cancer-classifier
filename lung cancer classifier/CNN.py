CNN
import tensorflow as tf
from tensorflow.keras import layers

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])
import tensorflow as tf
from tensorflow.keras import layers

# Assuming image_size is defined somewhere in your code
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(image_size, image_size),
    layers.Rescaling(1.0 / 255)
])
resize_and_rescale
from tensorflow.keras import models, layers, Input

# Define input shape
input_shape = (image_size, image_size, channels)
n_classes = 4

# Define input layer
inputs = Input(shape=input_shape)

# Define the model architecture
model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

# Add the batch dimension to the input shape
input_shape_with_batch = (batch_size,) + input_shape

# Build the model with the input shape including the batch dimension
model.build(input_shape=input_shape_with_batch)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
history = model.fit(train_ds,batch_size=batch_size,validation_data=val_ds,verbose=1,epochs=10)
scores = model.evaluate(test_ds)
