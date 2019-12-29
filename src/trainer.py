from __future__ import print_function

import itertools

import numpy as np

from os import mkdir
from os.path import isdir, join

from tqdm import tqdm

from totem import *

'''-------------------------------- PARAMS --------------------------------------'''
learning_rate = 0.001  # learning rate for training the model
l2_regularization = 0.0  # L2 regularization
lr_decay_factor = 10  # learning rate decay factor
lr_decay_iters = [20, 30, 35]  # Threshold for delta validation AUC score to decay lr
normalize = True  # normalize the data or not
normalization_axes = (0, 2)  # The axes for normalization
merge_tags = True  # merge the tags or not
n_tags = 50  # number of tags to use
batch_size = 32  # batch size
super_batch_size = 3000  # how much to send to the GPU in one go
train_size = 18000  # train dataset size
validation_size = 2000  # validation dataset size
seed = 1337  # the random seed
n_data_chunks = 8  # the number of chunks of the spectrograms
data_path = '../data'  # path to the processed spectrograms
n_iters = 40  # Number of iterations to run this for
'''------------------------------------------------------------------------------'''

rng = rng.RNG(seed)

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


def get_model(input_shape, n_outputs, rng):
    """
    Returns a generated model
    :param input_shape: The shape of the input that the model takes
    :param top: The number of outputs
    :param rng: An RNG object
    :return: The generated model
    """
    mscnn = model.Model(input_shape)
    mscnn.add_layer(layers.ConvLayer("Conv1.1", 50, (3, 7), rng, mode="half"))
    mscnn.add_layer(layers.PoolLayer("Pool1.1", (2, 4)))
    mscnn.add_layer(layers.ConvLayer("Conv1.2", 100, (3, 5), rng, mode="half"))
    mscnn.add_layer(layers.PoolLayer("Pool1.2", (2, 4)))
    mscnn.add_layer(layers.ConvLayer("Conv1.3", 70, (3, 3), rng, mode="half"))
    mscnn.add_layer(layers.PoolLayer("Pool1.3", (2, 2)))
    mscnn.add_layer(layers.PoolLayer("SubSample1", (2, 4), mode="avg"), source="inputs")
    mscnn.add_layer(layers.ConvLayer("Conv2.1", 100, (3, 5), rng, mode="half"))
    mscnn.add_layer(layers.PoolLayer("Pool2.1", (2, 4)))
    mscnn.add_layer(layers.ConvLayer("Conv2.2", 70, (3, 3), rng, mode="half"))
    mscnn.add_layer(layers.PoolLayer("Pool2.2", (2, 2)))
    mscnn.add_layer(layers.PoolLayer("SubSample2", (2, 4), mode="avg"), source="SubSample1")
    mscnn.add_layer(layers.ConvLayer("Conv3.1", 70, (3, 3), rng, mode="half"))
    mscnn.add_layer(layers.PoolLayer("Pool3.1", (2, 2)))
    mscnn.add_layer(layers.JoinLayer("Join1", axis=1), source=("Pool1.3", "Pool2.2", "Pool3.1"))
    mscnn.add_layer(layers.ConvLayer("ConvLast1", 70, (3, 3), rng, mode="half"))
    mscnn.add_layer(layers.ConvLayer("ConvLast2", 70, (3, 3), rng, mode="half"))
    mscnn.add_layer(layers.PoolLayer("PoolLast", (2, 2)))
    mscnn.add_layer(layers.FlattenLayer("FlattenLast"))
    mscnn.add_layer(layers.BNLayer("BNLast"))
    mscnn.add_layer(layers.DropOutLayer("Dropout1", rng, 0.6))
    mscnn.add_layer(layers.FCLayer("FCLast", 500, rng))
    mscnn.add_layer(layers.FCLayer("Top", n_outputs, rng, activation="sigmoid"))
    return mscnn


def remove_syn(tags, tag_names):
    flattened_syn = set(itertools.chain(*synonyms))
    tag_ids = [[tag_names.index(name)] for name in (set(tag_names) - flattened_syn)] + [list(map(tag_names.index,
                                                                                                 names))
                                                                                        for names in synonyms]
    new_tags = np.column_stack([np.logical_or.reduce(tags[:, tag], axis=1) for tag in tag_ids])
    return new_tags, [tag_names[x[0]] for x in tag_ids]


def load_data():
    tqdm.write('Loading data...')
    data_chunks = [np.load(join(data_path, 'spectrograms_{}.npy'.format(i))) for i in range(n_data_chunks)]
    data = np.concatenate(data_chunks, axis=0)
    del data_chunks
    if normalize:
        data -= data.mean(axis=normalization_axes, keepdims=True)
        data /= data.std(axis=normalization_axes, keepdims=True)
    tags = np.load(join(data_path, 'tags.npy'))
    tag_names = list(np.load(join(data_path, 'tag_names.npy')))
    if type(tag_names[0]) is not str:
        tag_names = [s.decode() for s in tag_names]
    if merge_tags:
        tags, tag_names = remove_syn(tags, tag_names)
    frequencies = tags.sum(axis=0)
    freq_order = np.argsort(frequencies)[::-1][:n_tags]
    tags = tags[:, freq_order]
    tag_names = np.array(tag_names)[freq_order]
    tqdm.write('Data loaded.')
    return data[:, None, :, :], tags, tag_names


def train(data, tags):
    if not isdir('../experiments'):
        mkdir('../experiments')
    data_size = len(data)
    test_size = data_size - (train_size + validation_size)
    assert test_size > 0, 'Not enough data'
    mscnn = get_model(data[0].shape, n_tags, rng)
    indices = np.arange(data_size)
    rng.shuffle(indices)
    validation_indices = indices[:validation_size]
    test_indices = indices[validation_size: validation_size + test_size]
    train_indices = indices[validation_size + test_size:]
    n_super_batches = (train_size - 1) // super_batch_size + 1
    super_batch_sizes = [min(train_size, (i + 1) * super_batch_size) - i * super_batch_size
                         for i in range(n_super_batches)]
    optimizer = optimizers.ADAM('bce', True, data[train_indices[:super_batch_sizes[0]]], tags[:super_batch_sizes[0]],
                                alpha=learning_rate, L2=l2_regularization)
    mscnn.build_optimizer(optimizer)
    n_batches = [(s - 1) // batch_size + 1 for s in super_batch_sizes]
    total_batches = sum(n_batches)
    validator = mscnn.get_runner(data[validation_indices], tags[validation_indices])
    tester = mscnn.get_runner(data[test_indices], tags[test_indices])
    mscnn.change_is_training(False)
    validation_aucs = [validator.auc_score(at_a_time=batch_size)]
    test_aucs = [tester.auc_score(at_a_time=batch_size)]
    for i in range(n_iters):
        bar = tqdm(total=total_batches, desc='Iteration {}:- Curr loss: NaN'.format(i + 1))
        mscnn.change_is_training(True)
        for j in range(n_super_batches):
            n_batch = n_batches[j]
            indices = train_indices[j * super_batch_size: (j + 1) * super_batch_size]
            optimizer.set_value(data[indices], tags[indices])
            order = np.arange(len(indices))
            for k in range(n_batch):
                loss = optimizer.train_step(order[k * batch_size: (k + 1) * batch_size])
                bar.set_description('Iteration {}:- Curr loss: {}'.format(i + 1, loss))
                bar.update()
        rng.shuffle(train_indices)
        mscnn.change_is_training(False)
        validation_score = validator.auc_score(at_a_time=batch_size)
        test_score = tester.auc_score(at_a_time=batch_size)
        tqdm.write('Validation AUC score: {}'.format(validation_score))
        tqdm.write('Test AUC score: {}'.format(test_score))
        test_aucs.append(test_score)
        if (i + 1) in lr_decay_iters:
            tqdm.write('Decaying learning rate by {}'.format(lr_decay_factor))
            optimizer.set_learning_rate(optimizer.get_learning_rate() / lr_decay_factor)
        if validation_score > np.max(validation_aucs):
            with open('../experiments/best_model.pkl', 'wb') as f:
                mscnn.save(f)
        validation_aucs.append(validation_score)
    tqdm.write('\n\nTest AUC score corresponding to best val score: {}'.format(test_aucs[np.argmax(validation_aucs)]))
    tqdm.write('Best test score: {}'.format(np.max(test_aucs)))


def main():
    data, tags, tag_names = load_data()
    train(data, tags)


if __name__ == '__main__':
    main()
