import sys
import time

from tensorflow.python.training.adam import AdamOptimizer

from audio.adapter import get_audio_adapter
from dataset import DatasetBuilder
from model import model_fn
from model.KerasUnet import getUnetModel
from utils.configuration import load_configuration
import tensorflow as tf
import csv

audio_path = './musdb_dataset/'
config_path = "./config/musdb_config.json"
INIT_LR = 1e-3
opt = AdamOptimizer(INIT_LR)
_instruments = ['vocals_spectrogram', 'bass_spectrogram', 'drums_spectrogram', 'other_spectrogram']
model_dict = {}
model_trainable_variables = {}

val_loss_results = []
val_metrics_results = []

export_dir = './spleeter_saved_model_dir/'
metrics_csv = './csv_metrics/metrics_loss.csv'

def get_training_dataset(audio_params, audio_adapter, audio_path):
    """ Builds training dataset.

    :param audio_params: Audio parameters.
    :param audio_adapter: Adapter to load audio from.
    :param audio_path: Path of directory containing audio.
    :returns: Built dataset.
    """
    builder = DatasetBuilder(
        audio_params,
        audio_adapter,
        audio_path,
        chunk_duration=audio_params.get('chunk_duration', 20.0),
        random_seed=audio_params.get('random_seed', 0))
    return builder.build(
        audio_params.get('train_csv'),
        cache_directory=audio_params.get('training_cache'),
        batch_size=audio_params.get('batch_size'),
        n_chunks_per_song=audio_params.get('n_chunks_per_song', 1),
        random_data_augmentation=False,
        convert_to_uint=True,
        wait_for_cache=False)


def get_validation_dataset(audio_params, audio_adapter, audio_path):
    """ Builds validation dataset.

    :param audio_params: Audio parameters.
    :param audio_adapter: Adapter to load audio from.
    :param audio_path: Path of directory containing audio.
    :returns: Built dataset.
    """
    builder = DatasetBuilder(
        audio_params,
        audio_adapter,
        audio_path,
        chunk_duration=20.0)
    return builder.build(
        audio_params.get('validation_csv'),
        batch_size=100,
        cache_directory=audio_params.get('validation_cache'),
        convert_to_uint=True,
        infinite_generator=False,
        n_chunks_per_song=1,
        # should not perform data augmentation for eval:
        random_data_augmentation=False,
        random_time_crop=False,
        shuffle=False,
    )

params = load_configuration(config_path)
audio_adapter = get_audio_adapter(None)

for instrument in _instruments:
    model_dict[instrument] = getUnetModel(instrument)
    model_trainable_variables[instrument] = model_dict[instrument].trainable_variables

def measureValAccuracy(test_features, test_label, run_ind):

    test_preds = {}

    for instrument in _instruments:
        test_preds[instrument] = model_dict[instrument].predict(test_features)

    test_losses = {
        name: tf.reduce_mean(tf.abs(output - test_label[name]))
        for name, output in test_preds.items()
    }
    test_loss = tf.reduce_sum(list(test_losses.values()))
    write_to_csv(test_loss, run_ind)
    print("[INFO] test loss: {:.4f}".format(test_loss))
    #metrics = {k: tf.keras.metrics.Mean(v) for k, v in test_losses.items()}
    #metrics['absolute_difference'] = tf.compat.v1.metrics.mean(test_loss)
    return test_loss



def write_to_csv(test_loss, run_ind):
    with open(metrics_csv, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([run_ind, test_loss])

def stepFn(inputFeatures, inputLabel):

    global opt
    output_dict = {}

    with tf.GradientTape(persistent=True) as tape:
        for instrument in _instruments:
            output_dict[instrument] = model_dict[instrument](inputFeatures['mix_spectrogram'])
        losses = {
            name: tf.reduce_mean(tf.abs(output - inputLabel[name]))
            for name, output in output_dict.items()
        }
        loss = tf.reduce_sum(list(losses.values()))
    for instrument in _instruments:
        grads = tape.gradient(loss, model_trainable_variables[instrument])
        opt.apply_gradients(zip(grads, model_trainable_variables[instrument]))

def saveIntermediateModel(save_dir, run_num):
    for instrument in _instruments:
        model_dict[instrument].save(save_dir + '/' + str(run_num) + '/' + instrument + '/')
        model_dict[instrument].save_weights(save_dir+'/' + str(run_num) + '/' + instrument + '/' + 'checkpoint' + '/')

input_ds = get_training_dataset(params, audio_adapter, audio_path )
test_ds = get_validation_dataset(params, audio_adapter, audio_path)


for run in range(1,11):
    sys.stdout.flush()
    runStart = time.time()
    print("[INFO] Run Step number is " + str(run))
    elem = next(iter(input_ds))
    input_features = elem[0]
    input_label = elem[1]
    stepFn(input_features, input_label)
    runEnd = time.time()
    elapsed = (runEnd - runStart)/ 60.0
    print("took {:.4} minutes".format(elapsed))

    if (run%5 == 0):
        test_elem = next(iter(test_ds))
        test_features = test_elem[0]
        test_label = test_elem[1]
        val_loss = measureValAccuracy(test_features, test_label, run)
        val_loss_results.append(val_loss)
        #val_metrics_results.append(val_metrics)

    if (run%10 == 0):
        saveIntermediateModel(export_dir, run)

print("model training completed")