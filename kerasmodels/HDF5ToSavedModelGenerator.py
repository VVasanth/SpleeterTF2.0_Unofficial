from tensorflow.keras.models import load_model

modelFile = "models/model_50.hdf5"
model = load_model(modelFile)

model.save("models/vocals_spectrogram")