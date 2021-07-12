import os
import numpy as np
import tensorflow as tf
from config import config


def load_captions_data(filename, split_char):
    with open(filename) as caption_file:
        caption_mapping = {}
        text_data = []
        caption_data = caption_file.readlines()

        for line in caption_data:
            line = line.rstrip("\n")
            img_name, caption = line.split(split_char)
            img_name = os.path.join(config['images_path'], img_name.strip())

            if img_name.endswith("jpg"):
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)
                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.9, shuffle=True):
    # 1. list of all image names
    all_images = list(caption_data.keys())

    if shuffle:
        np.random.shuffle(all_images)

    train_size = int(len(caption_data) * train_size)
    training_data = {img_name: caption_data[img_name] for img_name in all_images[:train_size]}
    validation_data = {img_name: caption_data[img_name] for img_name in all_images[train_size:]}

    return training_data, validation_data


def read_image(img_path, size=config['image_size']):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def make_dataset(images, captions, vectorization):
    img_dataset = tf.data.Dataset.from_tensor_slices(images).map(read_image,
                                                                 num_parallel_calls=config['autotune'])
    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(vectorization.vectorization,
                                                                   num_parallel_calls=config['autotune'])
    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(config['batch_size']).shuffle(256).prefetch(config['autotune'])
    return dataset
