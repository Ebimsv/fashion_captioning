import tensorflow as tf

config = dict(
    images_path="./images",
    image_size=(299, 299),
    seq_length=32,
    embed_dim=512,
    ff_dim=512,
    batch_size=32,
    epochs=8,
    autotune=tf.data.experimental.AUTOTUNE,
    dataset='description',
    annotations_dir="./annotations/",
    weights_dir="./weights/",
    tensorboard_logs_dir="./logs/",
    num_attention_heads=2,
)

