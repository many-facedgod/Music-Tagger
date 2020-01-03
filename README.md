# Music Tagging

This is the source code for the paper [A Multi-scale Convolutional Neural Network Architecture for Music Auto-Tagging, Dabral T.S., Deshmukh A.S., Malapati A.](https://link.springer.com/chapter/10.1007/978-981-13-1592-3_60) Our work aims to automatically tag the music clips in the MagnaTagATune dataset using a CNN architecture that takes into account the multiple temporal scales at which the musical features express themselves.

## Requirements
- Python 2
- Theano
- Librosa
- Numpy
- tqdm

We recommend the AWS AMI `ami-0231c1de0d92fe7a2`. Once the AMI is set up and the repository has been cloned, run the following commands to set up the environment:

    source activate theano_p27
    pip install tqdm
    pip install librosa
    sudo apt-get install libav-tools

The last command installs the codecs required to read the music files.

We make use of [Totem](https://github.com/many-facedgod/Totem), a library with a Theano backend that facilitates easy creation of feed-forward neural network. The library is a submodule for this git repository and so there is no need to install it separately.

## Getting the data

The MagnaTagATune dataset can be downloaded using the following command (in the `src`) directory:

    python get_data.py

This will download the data into the `data` folder and verify the downloads using its MD5.

## Preprocessing

We use the `librosa` library to preprocess the audio files into log-scaled mel-spectrograms. We use an FFT window size of 2048 and a stride length of 512. The sampling rate for the audio file is 11025. This extraction can be performed by running the following command:

    python gen_spectrograms.py

This launches 8 workers to convert the audio files into the spectrograms. The spectrograms are dumped into the `data` folder.

## Model Overview

Our model makes use of three subsampled versions of the spectrograms. A series of convolutions is run on all three versions of the spectrogram, and the three resultant tensors are concatenated depthwise before further convolutions and final prediction. The exact model can be found in the `get_model` function in `trainer.py`. 

## Training

We first merge the synonymous tags as suggested [here](https://github.com/keunwoochoi/magnatagatune-list). In particular, the list of synonymous tags is:

    synonyms = [['beat', 'beats'],
                ['chant', 'chanting'],
                ['choir', 'choral'],
                ['classical', 'clasical', 'classic'],
                ['drum', 'drums'],
                ['electro', 'electronic', 'electronica', 'electric'],
                ['fast', 'fast beat', 'quick'],
                ['female', 'female singer', 'female singing', 'female vocals', 'female voice', 'woman', 'woman singing',
                'women'],
                ['flute', 'flutes'],
                ['guitar', 'guitars'],
                ['hard', 'hard rock'],
                ['harpsichord', 'harpsicord'],
                ['heavy', 'heavy metal', 'metal'],
                ['horn', 'horns'],
                ['india', 'indian'],
                ['jazz', 'jazzy'],
                ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
                ['no beat', 'no drums'],
                ['no singer', 'no singing', 'no vocal', 'no vocals', 'no voice', 'no voices', 'instrumental'],
                ['opera', 'operatic'],
                ['orchestra', 'orchestral'],
                ['quiet', 'silence'],
                ['singer', 'singing'],
                ['space', 'spacey'],
                ['string', 'strings'],
                ['synth', 'synthesizer'],
                ['violin', 'violins'],
                ['vocal', 'vocals', 'voice', 'voices'],
                ['strange', 'weird']]

Our training set and validation set have 18000 and 2000 samples respectively. The remaining ~5800 samples are used as the test set.

We use the ADAM optimizer to optimize the weights of the neural network and train the network for 40 iterations. We start with a learning rate of 0.001 and decay it by a factor of ten at the 20th, 30th and the 35th epoch. Finally, we report the test AUC ROC score corresponding to the best validation score. The entire model is trained on the top 50 tags by frequency.

To run the training routine, run the command:

    python trainer.py

This will train the model with the given hyperparameters and will also save the best model in the `experiments` directory.

## Results

**Best Validation AUC-ROC score:** 0.904

**Corresponding test AUC-ROC score:** 0.900

## PyTorch implementation
For a recent PyTorch reimplemenation of the same model by Amala, check [here](https://github.com/amalad/Multi-Scale-Music-Tagger).

## Authors
- [Tanmaya Shekhar Dabral](https://github.com/many-facedgod)
- [Amala Sanjay Deshmukh](https://github.com/amalad)

## Citation
    @incollection{Dabral2018,
      doi = {10.1007/978-981-13-1592-3_60},
      url = {https://doi.org/10.1007/978-981-13-1592-3_60},
      year = {2018},
      month = dec,
      publisher = {Springer Singapore},
      pages = {757--764},
      author = {Tanmaya Shekhar Dabral and Amala Sanjay Deshmukh and Aruna Malapati},
      title = {A Multi-scale Convolutional Neural Network Architecture for Music Auto-Tagging},
      booktitle = {Advances in Intelligent Systems and Computing}
    }