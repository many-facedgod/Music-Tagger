# Music Tagging

This is the source code for the paper [A Multi-scale Convolutional Neural Network Architecture for Music Auto-Tagging](https://link.springer.com/chapter/10.1007/978-981-13-1592-3_60). Our work aims to automatically tag the music clips in the MagnaTagATune dataset using a CNN architecture that takes into account the multiple temporal scales at which the musical features express themselves.

## Training

The code uses [Totem](https://github.com/many-facedgod/Totem), a wrapper over Theano for easy creation and training of feedforward neural networks. The library should be cloned and installed before running this code. We suggest running the code on an AWS with the `ami-0231c1de0d92fe7a2` AMI, in the `theano_p27` conda environment. Note that Theano inherently has some stochasticity (even after seeding all RNGs) which seem to stem from cuDNN, and therefore the results may differ slightly from run to run, but always remain in the same ballpark.

Once the repository has been cloned, run:

    cd src
    python get_data.py
    python gen_spectrograms.py
    python trainer.py

The first script will download the data into `../data`, the second will extract the spectrograms and the third will train the model. The parameters for the scripts are clearly marked in the parameters section and can be changed.

Note that we also merge the synonymous tags as proposed [here](https://github.com/keunwoochoi/magnatagatune-list). This can be turned off in `trainer.py`.

## 

## Results

The expected test AUC score corresponding to the best validation score with the default parameters should be around 0.9.

## PyTorch implementation
For a recent PyTorch reimplemenation of the same model by Amala, check [here](https://github.com/amalad/Multi-Scale-Music-Tagger).

## Authors
- [Tanmaya Shekhar Dabral](https://github.com/many-facedgod)
- [Amala Sanjay Deshmukh](https://github.com/amalad)