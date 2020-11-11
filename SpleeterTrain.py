import sys
import time

from tensorflow.python.training.adam import AdamOptimizer

from audio.adapter import get_audio_adapter
from dataset import DatasetBuilder
from model import model_fn
from model.KerasUnet import getUnetModel
from utils.configuration import load_configuration
import tensorflow as tf

from functools import partial

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
        n_chunks_per_song=audio_params.get('n_chunks_per_song', 2),
        random_data_augmentation=False,
        convert_to_uint=True,
        wait_for_cache=False)

def _create_train_spec(params, audio_adapter, audio_path):
    """ Creates train spec.

    :param params: TF params to build spec from.
    :returns: Built train spec.
    """
    input_fn = partial(get_training_dataset, params, audio_adapter, audio_path)
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        max_steps=params['train_max_steps'])
    return train_spec



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
        chunk_duration=12.0)
    return builder.build(
        audio_params.get('validation_csv'),
        batch_size=audio_params.get('batch_size'),
        cache_directory=audio_params.get('validation_cache'),
        convert_to_uint=True,
        infinite_generator=False,
        n_chunks_per_song=1,
        # should not perform data augmentation for eval:
        random_data_augmentation=False,
        random_time_crop=False,
        shuffle=False,
    )


def _create_evaluation_spec(params, audio_adapter, audio_path):
    """ Setup eval spec evaluating ever n seconds

    :param params: TF params to build spec from.
    :returns: Built evaluation spec.
    """
    input_fn = partial(
        get_validation_dataset,
        params,
        audio_adapter,
        audio_path)
    evaluation_spec = tf.estimator.EvalSpec(
        input_fn=input_fn,
        steps=None,
        throttle_secs=params['throttle_secs'])
    return evaluation_spec

def _create_estimator(params):
    """ Creates estimator.

    :param params: TF params to build estimator from.
    :returns: Built estimator.
    """
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    model = getUnetModel()

    return model


audio_path = '/Users/vishrud/Desktop/Vasanth/Technology/Mobile-ML/Spleeter_TF2.0/musdb_dataset/sample'
config_path = "config/musdb_config.json"
params = load_configuration(config_path)
audio_adapter = get_audio_adapter(None)

train_spec = _create_train_spec(params, audio_adapter, audio_path)
evaluation_spec = _create_evaluation_spec(
    params,
    audio_adapter,
    audio_path)

INIT_LR = 1e-3
EPOCHS = 1
opt = AdamOptimizer(INIT_LR)
_instruments = ['vocals_spectrogram', 'bass_spectrogram', 'drums_spectrogram', 'other_spectrogram']
model_dict = {}
model_trainable_variables = {}

for instrument in _instruments:
    model_dict[instrument] = getUnetModel(instrument)
    model_trainable_variables[instrument] = model_dict[instrument].trainable_variables


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

input_ds = get_training_dataset(params, audio_adapter, audio_path )

for run in range(0,1000):
    sys.stdout.flush()
    epochStart = time.time()
    print("Run number is" + str(run))
    elem = next(iter(input_ds))
    input_features = elem[0]
    input_label = elem[1]
    stepFn(input_features, input_label)

export_dir = './spleeter_saved_model_dir/'

for instrument in _instruments:
    model_dict[instrument].save(export_dir + '/' + instrument + '/')

print("model training completed")