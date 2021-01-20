from pathlib import Path
import tensorflow as tf

vocals_model_dir = './kerasmodels/models/vocals_spectrogram'

vocals_converter = tf.lite.TFLiteConverter.from_saved_model(vocals_model_dir, signature_keys=['serving_default'])
vocals_converter.allow_custom_ops = True
vocals_converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
vocals_converter.target_spec.supported_types = [tf.uint8, tf.float32]
vocals_tflite_model = vocals_converter.convert()
with open('./tflite/vocals_model.tflite', 'wb') as f:
  f.write(vocals_tflite_model)

'''
other_converter = tf.lite.TFLiteConverter.from_saved_model(other_model_dir, signature_keys=['serving_default'])
other_converter.allow_custom_ops = True
other_converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
other_converter.target_spec.supported_types = [tf.uint8, tf.float32]
other_tflite_model = vocals_converter.convert()
with open('./tflite/other_model.tflite', 'wb') as f:
  f.write(vocals_tflite_model)


'''
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./tflite/vocals_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(100)