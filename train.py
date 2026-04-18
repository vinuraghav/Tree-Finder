import config, model_building, data_pipeline
from keras import Sequential
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = model_building.model_build()
x_test, y_test, x_train, y_train = data_pipeline.load_and_prep_data()


early_stopping = EarlyStopping(monitor=config.EARLY_STOPPING_MONITOR, patience=config.EARLY_STOPPING_PATIENCE)
model_checkpoint = ModelCheckpoint(filepath=f"{config.MODEL_CHECKPOINT_FP}.keras", monitor=config.MODEL_CHECKPOINT_MONITOR)
model.fit(x_train, y_train, epochs=config.EPOCHS, validation_data=(x_test, y_test), callbacks=[early_stopping, model_checkpoint])

