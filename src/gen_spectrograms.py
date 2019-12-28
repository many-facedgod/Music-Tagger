from __future__ import division
from __future__ import print_function

import warnings

import librosa
import numpy as np

from multiprocessing import Process
from os.path import join

from tqdm import tqdm

'''-------------------------------- PARAMS --------------------------------------'''
epsilon = 1e-20  # for log clipping
n_specs = None  # number of spectrograms required. None for all
sampling_rate = 11025  # sampling rate
base_path = '../data'  # The path of the dataset root
n_workers = 4  # The number of chunks to divide the dataset into
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


def worker_fn(filepaths, start_index, worker_id):
    """Function for generating one chunk of the data."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spectrograms = None
        size = None
        bar = tqdm(enumerate(filepaths, start_index), total=len(filepaths), position=worker_id)
        bar.set_description('Worker {}'.format(worker_id))
        for index, path in bar:
            if index in problematic:
                spectrograms[index - start_index] = np.zeros(size, dtype=np.float32)
                continue
            signal, rate = librosa.load(join(base_path, path), sr=sampling_rate)
            log_spectrogram = np.log(np.clip(librosa.feature.melspectrogram(signal, rate),
                                             epsilon, np.inf)).astype(np.float32)
            if index == start_index:
                size = log_spectrogram.shape
                spectrograms = np.empty((len(filepaths), size[0], size[1]), dtype=np.float32)
            spectrograms[index - start_index] = log_spectrogram
        if not worker_id:
            tqdm.write('\n\n\n\n\nMaximum log amplitude: {}'.format(spectrograms.max()))
            tqdm.write('Minimum log amplitude: {}'.format(spectrograms.min()))
            tqdm.write('Mean log amplitude: {}'.format(spectrograms.mean()))
            tqdm.write('Std. log amplitude: {}'.format(spectrograms.std()))
            tqdm.write('Spectrogram shape: {}'.format(size))
        np.save(join(base_path, 'spectrograms_{}.npy'.format(worker_id)), spectrograms)


def generate_spectrograms(filepaths):
    """Starts multiple workers to generate the spectrograms in parallel."""
    print('Generating spectrograms...')
    chunk_size = (len(filepaths) - 1) // n_workers + 1
    workers = [Process(target=worker_fn, args=(filepaths[i * chunk_size: (i + 1) * chunk_size],
                                               i * chunk_size, i)) for i in range(n_workers)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def main():
    ids, tags, filepaths, tag_names = read_tags()
    num_specs = min(n_specs, len(ids)) if n_specs is not None else len(ids)
    np.save(join(base_path, 'tags.npy'), tags[:num_specs])
    np.save(join(base_path, 'ids.npy'), ids[:num_specs])
    np.save(join(base_path, 'tag_names.npy'), tag_names)
    filepaths = filepaths[:num_specs]
    generate_spectrograms(filepaths)


if __name__ == '__main__':
    main()
