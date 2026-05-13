import json
import tensorflow as tf
from src.config import TRAIN_DIR, VALIDATION_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_DIR, MODEL_PATH
from src.model import build_transfer_model, build_cnn_model
from src.utils import save_class_names

def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError("Training dataset folder not found. Please add images inside dataset/train.")

    train_data = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    class_names = train_data.class_names
    save_class_names(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    # Change this to build_cnn_model(len(class_names)) if you want a normal CNN.
    model = build_transfer_model(len(class_names))

    model.summary()

    history = model.fit(
        train_data.map(lambda x, y: (data_augmentation(x, training=True), y)),
        validation_data=validation_data,
        epochs=EPOCHS
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    print(f"Model saved successfully at: {MODEL_PATH}")
    print(f"Classes: {class_names}")

if __name__ == "__main__":
    main()
