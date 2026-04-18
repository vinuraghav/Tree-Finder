import config
import os
import numpy as np
from PIL import Image

#THIS FILE DOES THE DATA PREPROCESSING

def load_and_prep_data():
    x = []
    y = []
    dataset_dir = config.DATASET_DIRECTORY
    
    for subfolder in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, f"{subfolder}")
        for items in os.listdir(file_path):
                image = Image.open(f"{file_path}/{items}")

                image = image.convert('RGB')
                image_resize = image.resize(config.IMG_SIZE)

                image_pixel_matrix = np.array(image_resize)

                x.append(image_pixel_matrix)
                y.append(config.LABELS[subfolder])

    x = np.array(x)
    y = np.array(y)    
    indices = np.arange(x.shape[0])


    np.random.shuffle(indices)


    x = x[indices]
    y = y[indices] 

    split_point = round(x.shape[0] * 0.8)


    x_train = np.array(x[0 : split_point])
    y_train = np.array(y[0 :  split_point])

    x_test = np.array(x[split_point::])
    y_test = np.array(y[split_point::]) 

    return x_test, y_test, x_train, y_train          


