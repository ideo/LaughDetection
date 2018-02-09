import tensorflow as tf
import keras
from datetime import datetime
import numpy as np
import tempfile
from scipy.io import wavfile

from audioset import vggish_embeddings
from laugh_detector.microphone_stream import MicrophoneStream

flags = tf.app.flags

flags.DEFINE_string(
    'keras_model', 'Models/LSTM_trained_on_audioset_40epoch.h5',
    'Path to trained keras model that will be used to run inference.')

flags.DEFINE_float(
    'sample_length', 3.0,
    'Length of audio sample to process in each chunk'
)

flags.DEFINE_string(
    'save_file', None,
    'Filename to save inference output to as csv. Leave empty to not save'
)

flags.DEFINE_bool(
    'print_output', True,
    'Whether to print inference output to the terminal'
)

flags.DEFINE_string(
    'recording_directory', None,
    'Directory where recorded samples will be saved'
    'If None, samples will not be saved'
)
FLAGS = flags.FLAGS

RATE = 16000
CHUNK = int(RATE * FLAGS.sample_length)  # 3 sec chunks


if __name__ == '__main__':
    model = keras.models.load_model(FLAGS.keras_model)
    audio_embed = vggish_embeddings.VGGishEmbedder()

    if FLAGS.save_file:
        writer = open(FLAGS.save_file, 'w')

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        for chunk in audio_generator:
            try:
                arr = np.frombuffer(chunk, dtype=np.int16)
                embeddings = audio_embed.convert_waveform_to_embedding(arr, RATE)
                p = model.predict(np.expand_dims(embeddings, axis=0))

                if FLAGS.recording_directory:
                    f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=FLAGS.recording_directory)
                    wavfile.write(f, RATE, arr)
                if FLAGS.print_output:
                    print(datetime.now().strftime('%H:%M:%S') + f' - Laugh Score: {p[0,0]:0.6f}')

                if FLAGS.save_file:
                    if FLAGS.recording_directory:
                        writer.write(datetime.now().strftime('%H:%M:%S') + f',{f.name},{p[0,0]}\n')
                    else:
                        writer.write(datetime.now().strftime('%H:%M:%S') + f',{p[0,0]}\n')

            except (KeyboardInterrupt, SystemExit):
                print('Shutting Down -- closing file')
                writer.close()
