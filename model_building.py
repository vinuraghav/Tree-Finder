import config
import data_pipeline
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomBrightness, RandomFlip, RandomRotation, RandomZoom


def model_build():
    model = Sequential()

    model.add(Input(shape=(*config.IMG_SIZE, config.INPUT_CHANNELS)))

    model.add(RandomFlip(seed=config.RANDOM_FLIP_SEED))
    model.add(RandomRotation(config.RANDOM_ROTATION))
    model.add(RandomZoom(config.RANDOM_ZOOM_HEIGHT, config.RANDOM_ZOOM_WIDTH))
    model.add(RandomBrightness(factor=config.RANDOM_BRIGHTNESS))

    #stacking inner layers
    model.add(Conv2D(32, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dropout(rate=config.DROPOUT_RATE))

    model.add(Dense(1, activation="sigmoid"))

    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy", "mse"])
    return model

