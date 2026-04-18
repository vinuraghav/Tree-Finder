LABELS = {
    "Banyan tree full shot": 0,
    "Banyan tree aerial roots": 0,
    "Banyan tree planted in park": 0,
    "Ficus benghalensis leaves": 0,
    "Banyan tree trunk close up":0,
    "Neem tree full shot": 1,
    "Azadirachta indica tree": 1,
    "Neem tree bark texture": 1,
    "Neem tree branches": 1,
    "Neem tree leaves close up": 1
}
IMG_SIZE = (128, 128)

DATASET_DIRECTORY = "tree_dataset"

EPOCHS = 1000

INPUT_CHANNELS = 3

#DROPOUT RATE
DROPOUT_RATE = 0.5
#Randomization of Inputs
RANDOM_ROTATION = (-0.2, 0.4)
RANDOM_ZOOM_HEIGHT = -0.4
RANDOM_ZOOM_WIDTH = -0.2
RANDOM_FLIP_SEED = 43
RANDOM_BRIGHTNESS = (-0.2, 0.2)

#TRAINING
EARLY_STOPPING_MONITOR = "val_accuracy"
EARLY_STOPPING_PATIENCE = 15

MODEL_CHECKPOINT_FP = r"best_instances/best_instance"
MODEL_CHECKPOINT_MONITOR = "val_accuracy"


#PREDICTION
MODEL_FILEPATH = r"best_instances\best_instance.keras"
PREDICTION_INPUT_IMAGE = r"images.jpg"
PREDICTION_INPUT_SIZE = (128, 128)