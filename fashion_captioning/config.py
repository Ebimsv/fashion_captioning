import tensorflow as tf

config = dict(
    images_path="./fashion_captioning/images/",
    image_size=(299, 299),
    seq_length=32,
    embed_dim=512,
    ff_dim=512,
    batch_size=32,
    epochs=80,
    autotune=tf.data.experimental.AUTOTUNE,
    dataset='title',
    annotations_dir="./fashion_captioning/annotations/",
    weights_dir="./fashion_captioning/weights/",
    tensorboard_logs_dir="./fashion_captioning/logs/",
    num_attention_heads=4,
)

