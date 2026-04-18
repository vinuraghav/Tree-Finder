#THIS IS SIMPLY A FILE THAT CAN BE RAN ON COMMAND PROMPT

#DIFFERENT WAY TO INPUT, SAME OUTPUT

from keras.saving import load_model
import config
from PIL import Image
import numpy as np
import sys

def input_image(filepath : str):
    prediction_input = Image.open(filepath)
    prediction_resize = prediction_input.resize(config.PREDICTION_INPUT_SIZE)
    prediciton_rgb = prediction_resize.convert("RGB")


    prediction_image_matrix = np.array(prediciton_rgb)

    prediction_processed = np.expand_dims(prediction_image_matrix, axis=0)

    return prediction_processed



model = load_model(config.MODEL_FILEPATH)
model.summary()

try:
    if __name__ == "__main__":
        filepath = sys.argv[1]
except IndexError:
    print("No Value found...please try again")
    sys.exit()

    

prediciton_image = input_image(filepath)

output = model.predict(prediciton_image)


if output[0][0] < 0.5:
    print("Its a Banyan Tree!")
elif output[0][0] > 0.5:
    print("Its a Neem Tree!")





