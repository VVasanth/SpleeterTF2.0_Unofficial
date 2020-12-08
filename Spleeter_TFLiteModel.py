from pathlib import Path
import tensorflow as tf

export_dir = './spleeter_saved_model_dir/'
subdirs = [x for x in Path(export_dir).iterdir()
           if x.is_dir() and 'temp' not in str(x)]
latest = str(sorted(subdirs)[-1])
converter = tf.lite.TFLiteConverter.from_saved_model(latest, signature_keys=['serving_default'])
converter.allow_custom_ops = True
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.uint8, tf.float32]
tflite_model = converter.convert()
with open('./tflite/model.tflite', 'wb') as f:
  f.write(tflite_model)


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./tflite/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(100)