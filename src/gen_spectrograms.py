from __future__ import division
from __future__ import print_function

import warnings

import librosa
import numpy as np

from os.path import join

from tqdm import tqdm

'''-------------------------------- PARAMS --------------------------------------'''
epsilon = 1e-20  # for log clipping
n_specs = None  # number of spectrograms required. None for all
sampling_rate = 11025  # sampling rate
base_path = '../data'  # The path of the dataset root
'''------------------------------------------------------------------------------'''

problematic = [16250, 24867, 25546]  # The files that cannot be opened


def read_tags():
    tag_file = open(join(base_path, 'tags.csv'))
    tags = []
    filepaths = []
    ids = []
    tag_names = [x.strip('\"') for x in tag_file.readline().strip().split('\t')[1:-1]]
    print('Number of tags: {}'.format(len(tag_names)))
    for line in tag_file:
        line = line.strip().split('\t')
        filepaths.append(line[-1].strip('\"'))
        ids.append(int(line[0].strip('\"')))
        tags.append([int(x.strip('\"')) for x in line[1:-1]])
    tags = np.array(tags)
    ids = np.array(ids)
    print('Number of data points: {}'.format(len(ids)))
    return ids, tags, filepaths, tag_names


def generate_spectrograms(filepaths):
    print('Generating Spectrograms...')
    spectrograms = None
    size = None
    for index, path in tqdm(enumerate(filepaths), total=len(filepaths)):
        if index in problematic:
            spectrograms[index] = np.zeros(size, dtype=np.float32)
            continue
        signal, rate = librosa.load(join(base_path, path), sr=sampling_rate)
        log_spectrogram = np.log(np.clip(librosa.feature.melspectrogram(signal, rate), epsilon, np.inf)).astype(np.float32)
        if not index:
            size = log_spectrogram.shape
            spectrograms = np.empty((len(filepaths), size[0], size[1]), dtype=np.float32)
        spectrograms[index] = log_spectrogram
    print('Spectrograms generated')
    print('Maximum log amplitude: {}'.format(spectrograms.max()))
    print('Minimum log amplitude: {}'.format(spectrograms.min()))
    print('Mean log amplitude: {}'.format(spectrograms.mean()))
    print('Std. log amplitude: {}'.format(spectrograms.std()))
    print('Spectrogram shape: {}'.format(size))
    return spectrograms


def main():
    ids, tags, filepaths, tag_names = read_tags()
    num_specs = min(n_specs, len(ids)) if n_specs is not None else len(ids)
    np.save(join(base_path, 'tags.npy'), tags[:num_specs])
    np.save(join(base_path, 'ids.npy'), ids[:num_specs])
    np.save(join(base_path, 'tag_names.npy'), tag_names)
    filepaths = filepaths[:num_specs]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spectrograms = generate_spectrograms(filepaths)
        print('Dumping spectrograms...')
        np.save(join(base_path, 'spectrograms.npy'), spectrograms)


if __name__ == '__main__':
    main()
