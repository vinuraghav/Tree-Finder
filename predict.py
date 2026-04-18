from keras.saving import load_model
import config
from PIL import Image
import numpy as np
import sys

def input_image():
    prediction_input = Image.open(config.PREDICTION_INPUT_IMAGE)
    prediction_resize = prediction_input.resize(config.PREDICTION_INPUT_SIZE)
    prediciton_rgb = prediction_resize.convert("RGB")


    prediction_image_matrix = np.array(prediciton_rgb)

    prediction_processed = np.expand_dims(prediction_image_matrix, axis=0)

    return prediction_processed

model = load_model(config.MODEL_FILEPATH)
model.summary()
prediciton_image = input_image()

output = model.predict(prediciton_image)[0][0]


if output < 0.5:
    print(f"Its a Banyan Tree!...{round(output, 4)}")
elif output > 0.5:
    print(f"Its a Neem Tree!..{round(output, 4)}")





