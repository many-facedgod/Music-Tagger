from __future__ import print_function
import numpy as np
import theano
import sys
base_path = "/datasets/spectrogram"  # where everything is.
sys.path.append(base_path)
from Totem import *
import gzip
import pickle
import os
import time
import model2

####PARAMS######
top = 50  # Use only the top "x" labels, frequency wise
delete = [16250, 24867, 25546]  # the problematic ones
spec_file_name = "Spectrograms.npy"  # the filename of the spectrograms file
learning_rate = 0.15  # the learning rate parameter
batch_size = 20  # the batch size for training
momentum = 0.03  # the momentum
seed = 12345  # the seed for the RNG class in Totem
Train_sizes = [5000, 5000, 5000, 5000]
""" Trains_sizes is the list of "super batches" that are to be put into the GPU memory one at a time."""
Validation_size = 5800  # Validation size
model_input_shape = (1, 128, 628)  # the input shape to the model (one instance).
epochs = 100
at_a_time = 20  # How many to be run at a time during the validation phase
assert np.sum(Train_sizes) + Validation_size <= 25860, "Not enough training data."
#################

tag_freqs, tag_names, vectors = pickle.load(gzip.open(os.path.join(base_path, "TuneAnnotationsNoSyn.pkl.gz"), "rb"),
                                            encoding='latin1')
spectrograms = np.load(open(os.path.join(base_path, spec_file_name), 'rb'))
rng = rng.RNG(seed)  # Note: Even though I seed everything, there is some stochasticity going on. No clue why.

##########Processing the data#################
vectors = np.delete(vectors, delete, axis=0)
spectrograms = np.delete(spectrograms, delete, axis=0)
shuffle_inds = np.arange(vectors.shape[0])
rng.shuffle(shuffle_inds)  # Shuffling the data
vectors = vectors[shuffle_inds][:, :top].astype(theano.config.floatX)
spectrograms = np.expand_dims(spectrograms[shuffle_inds], axis=1)  # Extra axis for the channel
##############################################
model = model2.get_model(model_input_shape, top, rng)
#model = pickle.load(open("TrainedModel.pkl.gz", "rb"))
train_indices = np.arange(int(np.sum(Train_sizes)))
rng.shuffle(train_indices)
###################################Building the optimizer##############################################
optimizer = optimizers.ADAM("bce", True, spectrograms[train_indices[:Train_sizes[0]]],
                            vectors[train_indices[:Train_sizes[0]]], L2=0.0001)  # BCE is binary cross entropy
model.build_optimizer(optimizer)
######################################################################################################

total_train = np.sum(Train_sizes, dtype=np.int32)
validator = model.get_runner(spectrograms[total_train: total_train + Validation_size],
                             vectors[total_train: total_train + Validation_size])
best = 0.0  # best validation score
nbatches = [int(np.ceil(sizes / float(batch_size))) for sizes in Train_sizes]
start = time.time()
logfile=open("Training.log", "w")
for i in range(epochs):
    start_iter = time.time()
    error = []
    model.change_is_training(True)
    for ind, train_size in enumerate(Train_sizes):
        pad = np.sum(Train_sizes[:ind], dtype=np.int32)
        optimizer.set_value(spectrograms[train_indices[pad: pad + train_size]],
                            vectors[train_indices[pad: pad + train_size]])
        indices = np.arange(train_size)
        for j in range(nbatches[ind]):
            error.append(optimizer.train_step(indices[j * batch_size: (j + 1) * batch_size]))
    rng.shuffle(train_indices)
    print("Iteration {}".format(i))
    print("Training cost is: {}".format(np.mean(error)))
    model.change_is_training(False)
    valid_score = validator.auc_score(at_a_time=at_a_time)
    print("Validation AUC score is: {}".format(valid_score))
    if valid_score > best:
        best = valid_score
        model.save(gzip.open("TrainedModel.pkl.gz", "wb"))
    end_iter = time.time()
    print("Time for {} iteration: {} seconds".format(i, end_iter - start_iter))
    logfile.write(str(i)+" "+str(np.mean(error))+" "+str(valid_score)+" "+str(end_iter-start_iter)+"\n")

end = time.time()
logfile.close()
print("Best validation score: {}".format(best))
print("Total time taken is {} seconds".format(end - start))