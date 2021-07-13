import tensorflow as tf

config = dict(
    images_path="./images",
    image_size=(299, 299),
    seq_length=32,
    embed_dim=512,
    ff_dim=512,
    batch_size=32,
    epochs=8,
    autotune=tf.data.experimental.AUTOTUNE
)

config_title = dict(
    train_annotations_file="./annotations/train_title.txt",
    test_annotations_file='./annotations/test_title.txt',
    split_char='\t',
    weights_path="./weights/title/2_heads/",
    tensorboard_logs_dir="./logs/title/2_heads",
    num_attention_heads=2,
)
config_title.update(config)

config_description = dict(
    train_annotations_file="./annotations/train_description.txt",
    test_annotations_file='./annotations/test_description.txt',
    split_char=',',
    weights_path="./weights/description/2_heads/",
    tensorboard_logs_dir="./logs/description/2_heads",
    num_attention_heads=2,
)
config_description.update(config)
