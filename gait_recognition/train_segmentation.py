# Install required packages
#!pip install -q albumentations

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Create DataFrame with image and mask paths
import os

image_dir = '/content/drive/MyDrive/segmentation/images'
mask_dir = '/content/drive/MyDrive/segmentation/masks'

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Create DataFrame
df = pd.DataFrame({
    'images': [os.path.join(image_dir, f) for f in image_files],
    'masks': [os.path.join(mask_dir, f) for f in image_files]
})

# Display sample image and mask
row = df.iloc[0]
image_path = row['images']
mask_path = row['masks']

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('GROUND TRUTH')
ax2.imshow(mask, cmap='gray')

# Constants
IMG_SIZE = (128, 128)

def load_and_preprocess(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # normalize to [0,1]

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)  # 1 channel for mask
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = mask / 255.0  # normalize mask too

    return img, mask

# Split dataset
image_paths = df['images'].values
mask_paths = df['masks'].values

train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42)

# Training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Data augmentation
transform = A.Compose([
    A.Resize(128, 128),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
])

class Dataset(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=16, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        X, Y = [], []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            img = cv2.imread(self.image_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)

            if self.augment:
                augmented = transform(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']

            img = img / 255.0
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)

# U-Net model architecture
def conv_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(inputs, filters):
    x = conv_block(inputs, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape, num_classes):
    inputs = layers.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid")(d4)

    return Model(inputs, outputs, name="U-Net")

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# Create and compile model
model = build_unet(input_shape=(128, 128, 3), num_classes=1)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', iou_metric])

# Create a directory to save models if it doesn't exist
os.makedirs('/content/drive/MyDrive/segmentation/models', exist_ok=True)

# Create ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    '/content/drive/MyDrive/segmentation/models/model_epoch_{epoch:02d}.h5',
    save_best_only=False,  # Save every epoch
    save_weights_only=False,  # Save the entire model
    monitor='val_loss',  # Monitor validation loss
    verbose=1  # Print a message when a model is saved
)

# Train model with checkpoint callback
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,
    batch_size=16,
    callbacks=[checkpoint]
)

# Plot training history
plt.figure(figsize=(12, 6))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# IoU curve
plt.subplot(1, 2, 2)
plt.plot(history.history['iou_metric'], label='Train IoU')
plt.plot(history.history['val_iou_metric'], label='Val IoU')
plt.title("IoU Curve")
plt.xlabel("Epochs")
plt.ylabel("IoU")
plt.legend()

plt.tight_layout()
plt.show()

# Visualize predictions
for images, masks in val_dataset.take(1):
    sample_image = images[2]
    sample_mask = masks[2]
    break
    
pred_mask = model.predict(tf.expand_dims(sample_image, axis=0))[0]
pred_mask = tf.cast(pred_mask > 0.5, tf.float32)

plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(sample_image)
plt.title("Original Image")
plt.axis("off")

# Ground Truth Mask
plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(sample_mask), cmap='gray')
plt.title("Ground Truth Mask")
plt.axis("off")

# Predicted Mask
plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(pred_mask), cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show() 