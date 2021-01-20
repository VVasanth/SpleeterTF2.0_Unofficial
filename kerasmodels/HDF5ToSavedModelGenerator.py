from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf

def custom_loss(y_actual, y_predicted):
	custom_loss_value = K.square(y_predicted-y_actual)
	custom_loss_value = K.sum(custom_loss_value, axis=1)
	return custom_loss_value

def dice_coefficient(y_true, y_pred):
	numerator = 2 * tf.reduce_sum(y_true * y_pred)
	denominator = tf.reduce_sum(y_true + y_pred)
	return numerator / (denominator + tf.keras.backend.epsilon())

modelFile = "models/vocal_model_300epoch.hdf5"
model = load_model(modelFile, custom_objects={"loss":custom_loss, "metrics":dice_coefficient}, compile=False)

model.save("models/vocals_spectrogram")