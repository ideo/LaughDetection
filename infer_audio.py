import numpy as np
import tensorflow as tf
import glob

from audioset import vggish_embeddings

import keras

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'wav_directory', None,
    'Path to a directory of wav files for inference.'
    'Predictions will be written to predict.csv in that directory'
)
flags.DEFINE_string(
    'keras_model', 'Models/LSTM_trained_on_audioset_40epoch.h5',
    'Path to trained keras model that will be used to run inference.')


flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def predict_laugh(processed_embedding):
    model = keras.models.load_model(FLAGS.keras_model)
    return model.predict(processed_embedding)


if __name__ == '__main__':
    audio_embedder = vggish_embeddings.VGGishEmbedder(FLAGS.tfrecord_file)

    if FLAGS.wav_directory:
        files = glob.glob(FLAGS.wav_directory+'/*.wav')
        embeddings = [audio_embedder.convert_audio_to_embedding(f) for f in files]
        max_len = np.max([e.shape[0] for e in embeddings])
        embeddings = np.array([np.append(e, np.zeros([(max_len - e.shape[0]), 128], np.float32), axis=0) for e in embeddings])
        scores = predict_laugh(embeddings)
        for name, score in zip(files, scores[:, 0]):
            print('{:>12}:  {:0.6f}'.format(name, score))

    else:
        processed_embedding =  audio_embedder.convert_audio_to_embedding(FLAGS.wav_file)
        p = predict_laugh(np.expand_dims(processed_embedding, axis=0))
        print('Laugh Score: {}'.format(p))