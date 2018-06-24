# Laugh Detector

This is the code accompanying blog posts (Here and Here) for training and running a laugh detector. This project is based on [AudioSet](https://research.google.com/audioset/) and includes a rebalanced subset of their data and utilizes their pre-trained audio feature vectorizer [vggish](https://github.com/tensorflow/models/tree/master/research/audioset). I also followed the lead of a [similar project](https://github.com/ganesh-srinivas/laughter) for processing the AudioSet data and choosing network architectures

This repo contains code for running live inference of the presence of laughter, a few pre-trained models, code that was used for training, a subset of the AudioSet data containing a balanced set of laughter and non-laughter human speech and a dashboard for visualizing the amount of laughter.

## Running Live Inference

Running `live_inference.py` will use your computer's microphone to capture audio and infer the presence of laughter in segmented chunks. These scores can be written to a time-stamped .csv and/or used to control a wireless light on a Philips Hue bridge.

### Setup

First, install the python requirements:
`pip install -r requirements.txt`

You will then need to download [vggish_model.ckpt](https://storage.googleapis.com/audioset/vggish_model.ckpt) from AudioSet and move it to the `audioset` directory to run the vggish model.

You will also need to install [Portaudio](http://www.portaudio.com/download.html) to run the live inference script. On macOS you can accomplish this with [homebrew](https://brew.sh/)

`$ brew install portaudio`

### Running the Detector

You can run the laugh detector using a pre-trained model from the command line

`$ python live_inference.py --save_file='\path\to\output.csv'`

If you want to control a Philips Hue bulb, first get the IP address of your bridge, then run

`$ python live_inference.py --hue_lights='True' --hue_IP='your.bridge.ip.address'`

The first time you run this, you will need to press the connection button within 30 seconds of the phue library loading. See the [phue library](https://github.com/studioimaginaire/phue) if you have trouble connecting

## Using the Dashboard

The dashboard is built using [Dash](https://dash.plot.ly/), you can run it with

`$ python dashboard/dashboard.py`

It is currently working on a sample data set, but you can edit the file to point to a new datafile.

## Training Your Own Models

This repo contains three pre-trained models in the Models directory. You can run the code in `Notebooks/ModelTraining.ipynb` to repeat the training process or to use it as a skeleton for training your own models. This notebook uses [keras-tqdm](https://github.com/bstriner/keras-tqdm) for the progress bars, which doesn't play well with JupyterLab. You can either run the notebook in a classic Jupyter Notebook or disable keras-tqdm, it won't change the training process.

This repo contains two processed subsets of the AudioSet data: a training set of 18,768 samples where half the data has a laughter label (including "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", and "Chuckle, chortle") and the other half has a non-laughter human noise label (including "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Babbling", "Crowd", and "Hubbub, speech noise, speech babble"). Samples can have multiple labels, but a positive "laughter" label is only assigned if one of the laughter labels is present and a negative label affirms the absence of any laughter label. The evaluation set contains 568 samples from the evaluation data with the same distribution. These rebalanced datasets are in `Data/bal_laugh_speech_subset.tfrecord` and `Data/eval_laugh_speech_subset.tfrecord` respectively.

The pre-trained models and re-balanced subset data are all licensed under a [Creative Commons Attribution 4.0 International License.](https://creativecommons.org/licenses/by/4.0/)

If you want to create your own processed subset of the AudioSet data on a different collection of labels, you can follow the code in `Notebooks/AudiosetProcessing.ipynb`. You can create your own subsets of the Audioset data by creating a list of labels to use for the positive and negative cases.




