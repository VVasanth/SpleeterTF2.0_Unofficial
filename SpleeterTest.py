from librosa import istft, stft
from scipy.signal.windows import hann
from tensorflow.python.keras.layers import Multiply

from audio.adapter import get_audio_adapter
import numpy as np
import tensorflow as tf

from utils.tensor import pad_and_partition
import os
from os.path import basename, join, splitext, dirname

from multiprocessing import Pool

multiprocess=True
audio_adapter = get_audio_adapter(None)
audio_descriptor = '/Users/vishrud/Desktop/Vasanth/Technology/Mobile-ML/Spleeter_TF2.0/input/Actions_DevilsWords.wav'
sample_rate = 44100
_instruments = ['vocals_spectrogram', 'other_spectrogram']
_pool = Pool() if multiprocess else None
_tasks = []
destination = '/Users/vishrud/Desktop/Vasanth/Technology/Mobile-ML/Spleeter_TF2.0/output/'
_sample_rate = 44100

export_dir = './spleeter_saved_model_dir/2000_2511/'
EPSILON = 1e-10
_frame_length = 4096
_F = 1024

def _stft(data, inverse=False, length=None):
    """
    Single entrypoint for both stft and istft. This computes stft and istft with librosa on stereo data. The two
    channels are processed separately and are concatenated together in the result. The expected input formats are:
    (n_samples, 2) for stft and (T, F, 2) for istft.
    :param data: np.array with either the waveform or the complex spectrogram depending on the parameter inverse
    :param inverse: should a stft or an istft be computed.
    :return: Stereo data as numpy array for the transform. The channels are stored in the last dimension
    """
    assert not (inverse and length is None)
    data = np.asfortranarray(data)
    N = 4096
    H = 1024
    win = hann(N, sym=False)
    fstft = istft if inverse else stft
    win_len_arg = {"win_length": None, "length": length} if inverse else {"n_fft": N}
    n_channels = data.shape[-1]
    out = []
    for c in range(n_channels):
        d = data[:, :, c].T if inverse else data[:, c]
        s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
        s = np.expand_dims(s.T, 2 - inverse)
        out.append(s)
    if len(out) == 1:
        return out[0]
    return np.concatenate(out, axis=2 - inverse)




def _extend_mask( mask):
    """ Extend mask, from reduced number of frequency bin to the number of
    frequency bin in the STFT.

    :param mask: restricted mask
    :returns: extended mask
    :raise ValueError: If invalid mask_extension parameter is set.
    """
    extension = "zeros"
    # Extend with average
    # (dispatch according to energy in the processed band)
    if extension == "average":
        extension_row = tf.reduce_mean(mask, axis=2, keepdims=True)
    # Extend with 0
    # (avoid extension artifacts but not conservative separation)
    elif extension == "zeros":
        mask_shape = tf.shape(mask)
        extension_row = tf.zeros((
            mask_shape[0],
            mask_shape[1],
            1,
            mask_shape[-1]))
    else:
        raise ValueError(f'Invalid mask_extension parameter {extension}')
    n_extra_row = _frame_length // 2 + 1 - _F
    extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
    return tf.concat([mask, extension], axis=2)


def maskOutput(output_dict, stft_val):
    separation_exponent = 2
    output_sum = tf.reduce_sum(
        [e ** separation_exponent for e in output_dict.values()],
        axis=0
    ) + 1e-10
    out = {}
    for instrument in _instruments:
        output = output_dict[f'{instrument}_spectrogram']
        # Compute mask with the model.
        instrument_mask = (output ** separation_exponent
                           + (EPSILON / len(output_dict))) / output_sum
        # Extend mask;
        instrument_mask = _extend_mask(instrument_mask)
        # Stack back mask.
        old_shape = tf.shape(instrument_mask)
        new_shape = tf.concat(
            [[old_shape[0] * old_shape[1]], old_shape[2:]],
            axis=0)
        instrument_mask = tf.reshape(instrument_mask, new_shape)
        # Remove padded part (for mask having the same size as STFT);

        instrument_mask = instrument_mask[
                          :tf.shape(stft_val)[0], ...]
        out[f'{instrument}_spectrogram'] = instrument_mask

    for instrument, mask in out.items():
        out[instrument] = tf.cast(mask, dtype=tf.complex64) * stft_val  # --> updating code locally
        print(instrument)
        print(tf.reduce_sum(out[instrument]))
        #out[instrument] = tf.cast(mask, dtype=tf.complex64)

    return out


def join_self(timeout=200):
    """ Wait for all pending tasks to be finished.

    :param timeout: (Optional) task waiting timeout.
    """
    while len(_tasks) > 0:
        task = _tasks.pop()
        task.get()
        task.wait(timeout=timeout)


def save_to_file(
        sources, audio_descriptor, destination,
        filename_format='{filename}/{instrument}.{codec}',
        codec='wav', audio_adapter = None,
        bitrate='128k', synchronous=True):
    """ export dictionary of sources to files.

    :param sources:             Dictionary of sources to be exported. The
                                keys are the name of the instruments, and
                                the values are Nx2 numpy arrays containing
                                the corresponding intrument waveform, as
                                returned by the separate method
    :param audio_descriptor:    Describe song to separate, used by audio
                                adapter to retrieve and load audio data,
                                in case of file based audio adapter, such
                                descriptor would be a file path.
    :param destination:         Target directory to write output to.
    :param filename_format:     (Optional) Filename format.
    :param codec:               (Optional) Export codec.
    :param audio_adapter:       (Optional) Audio adapter to use for I/O.
    :param bitrate:             (Optional) Export bitrate.
    :param synchronous:         (Optional) True is should by synchronous.

    """

    foldername = basename(dirname(audio_descriptor))
    filename = splitext(basename(audio_descriptor))[0]
    generated = []
    for instrument, data in sources.items():
        path = join(destination, filename_format.format(
                filename=filename,
                instrument=instrument,
                foldername=foldername,
                codec=codec,
                ))
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if path in generated:
            raise Exception((
                f'Separated source path conflict : {path},'
                'please check your filename format'))
        generated.append(path)
        if _pool:
            task = _pool.apply_async(audio_adapter.save, (
                path,
                data,
                _sample_rate,
                codec,
                bitrate))
            _tasks.append(task)
        else:
            audio_adapter.save(path, data, _sample_rate, codec, bitrate)
    if synchronous and _pool:
        join_self()


def separate(waveform, audio_descriptor):
    stft_val = _stft(waveform)
    if stft_val.shape[-1] == 1:
        stft_val = np.concatenate([stft_val, stft_val], axis=-1)
    elif stft_val.shape[-1] > 2:
        stft_val = stft_val[:, :2]

    spectrogram = tf.abs(pad_and_partition(stft_val, 512))[:,:,:1024,:]

    stft_tf = tf.convert_to_tensor(stft_val,np.float32)

    #self._features[spec_name] = tf.abs(
     #   pad_and_partition(self._features[stft_name], self._T))[:, :, :self._F, :]

    preds = {}

    for instrument in _instruments:
        predict_model = tf.saved_model.load(export_dir + instrument)
        inference_func = predict_model.signatures["serving_default"]
        predictions = inference_func(spectrogram)
        preds[f'{instrument}_spectrogram'] = predictions[f'{instrument}_spectrogram']


    #preds1 = {}
    #preds1['vocals_spectrogram_spectrogram'] = spectrogram
    input_RedSum = tf.reduce_sum(spectrogram)
    outputRedSum = tf.reduce_sum(preds['vocals_spectrogram_spectrogram'])
    output_dict = maskOutput(preds, stft_val)
    #output_dict_1 = maskOutput(preds1, stft_val)

    #input_RedSum1 = tf.reduce_sum(output_dict_1['vocals_spectrogram_spectrogram'])
    output_RedSum1 = tf.reduce_sum(output_dict['vocals_spectrogram_spectrogram'])

    out = {}
    for instrument in _instruments:
        out[instrument] = _stft(output_dict[f'{instrument}_spectrogram'], inverse=True, length=waveform.shape[0])

    filename_format = '{filename}/{instrument}.{codec}'
    codec = 'wav'
    bitrate = '128k'
    synchronous = False
    save_to_file(out, audio_descriptor, destination,
                      filename_format, codec, audio_adapter,
                      bitrate, synchronous)

    print(100)

waveform, sample_rate = audio_adapter.load(
            audio_descriptor,
            offset=0,
            duration=600,
            sample_rate=44100)

sources = separate(waveform, audio_descriptor)

