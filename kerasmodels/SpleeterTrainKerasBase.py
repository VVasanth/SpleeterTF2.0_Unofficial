from tensorflow.python.training.adam import AdamOptimizer

from audio.adapter import get_audio_adapter
from dataset import DatasetBuilder
from kerasmodels.EpochCheckpoint import EpochCheckpoint
from kerasmodels.TrainingMonitor import TrainingMonitor
from model.KerasUnet import getUnetModel
from utils.configuration import load_configuration
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

audio_path = '../musdb_dataset/'
config_path = "../config/musdb_config.json"
INIT_LR = 1e-3
opt = AdamOptimizer(INIT_LR)
_instruments = ['vocals']
model_dict = {}
model_trainable_variables = {}

val_loss_results = []
val_metrics_results = []

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
        "../config/musdb_train_small.csv",
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
        "../config/musdb_validation_small.csv",
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

# construct the argument parse and parse the arguments
checkPointVal = './kerasmodels/models'
modelVal = None
startEpochVal = 120
INIT_LR = 1e-3


def custom_loss(y_actual, y_predicted):
	custom_loss_value = K.square(y_predicted-y_actual)
	custom_loss_value = K.sum(custom_loss_value, axis=1)
	return custom_loss_value

def dice_coefficient(y_true, y_pred):
	numerator = 2 * tf.reduce_sum(y_true * y_pred)
	denominator = tf.reduce_sum(y_true + y_pred)
	return numerator / (denominator + tf.keras.backend.epsilon())


def trainModelOverEpochs():

	# if there is no specific model checkpoint supplied, then initialize
	# the network and compile the model
	if modelVal is None:
		print("[INFO] compiling model...")
		opt = AdamOptimizer(INIT_LR)
		model = getUnetModel("vocals")
		model.compile(loss=custom_loss, optimizer=opt,
			metrics=[dice_coefficient])
	# otherwise, we're using a checkpoint model
	else:
		# load the checkpoint from disk
		print("[INFO] loading {}...".format(modelVal))
		model = load_model(modelVal)

		# update the learning rate
		print("[INFO] old learning rate: {}".format(
			K.get_value(model.optimizer.lr)))
		K.set_value(model.optimizer.lr, 1e-3)
		print("[INFO] new learning rate: {}".format(
			K.get_value(model.optimizer.lr)))


	# build the path to the training plot and training history
	plotPath = "./kerasmodels/plots/resnet_fashion_mnist.png"
	jsonPath = "./kerasmodels/plots/resnet_fashion_mnist.json"

	# construct the set of callbacks
	callbacks = [
		EpochCheckpoint(checkPointVal, every=5,
			startAt=startEpochVal),
		TrainingMonitor(plotPath,
			jsonPath=jsonPath,
			startAt=startEpochVal)]


	input_ds = get_training_dataset(params, audio_adapter, audio_path )
	test_ds = get_validation_dataset(params, audio_adapter, audio_path)

	model.fit_generator(
		input_ds,
		validation_data=test_ds,
		steps_per_epoch=64,
		epochs=50,
		callbacks=callbacks,
		verbose=1)

	print(100)