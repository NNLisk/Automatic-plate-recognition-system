import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

img_size = (64, 64)
batch = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/OCR_training_data/data/train",
    image_size=img_size,
    batch_size=batch,
    label_mode="categorical",
    shuffle=True
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/OCR_training_data/data/test",
    image_size=img_size,
    batch_size=batch,
    label_mode="categorical",
    shuffle=False
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/OCR_training_data/data/val",
    image_size=img_size,
    batch_size=batch,
    label_mode="categorical",
    shuffle=False
)