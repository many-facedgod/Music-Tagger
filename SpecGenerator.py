from __future__ import print_function
import numpy as np
import gzip
import cPickle as pickle
import os
import librosa
import time
import logging

# PARAMS:
epsilon = 1e-20  # for log clipping
log = np.log  # the log function
n = 1000  # number of spectrograms required. Total = 25863
sr = 11025  # sampling rate
base_path = "/media/tanmaya/01CFA9F3FE6BF9F0/Downloads/MagnaTagATune"  # The path of the dataset root
_type = np.float32  # The dtype of the spectrogram arrays
print_every = 10  # Print a message after every few spectrograms
use_numpy = True  # Use numpy for serialization or not
compress = False  # Save in a compressed file
log_file = "SpecGen.log"
problematic = [16250, 24867, 25546]  # The files that cannot be opened
"""
Using numpy is highly recommended as it offers much faster dumping times as well lower file sizes.
Compressed numpy files are approximately 10x slower to dump than uncompressed ones. The size is approximately 90% of the
uncompressed numpy file.
Recommended: numpy uncompressed

"""
logging.basicConfig(filename=os.path.join(base_path, log_file), level=logging.INFO, format="%(message)s")
logging.info("***************************NEW RUN****************************")
logging.info("\nPARAMS:")
logging.info(
    "epsilon: {}\nn: {}\nsr: {}\nbase_path: {}\ntype: {}\nuse_numpy: {}\ncompress: {}".format(epsilon, n, sr, base_path,
                                                                                              _type, use_numpy,
                                                                                              compress))
clip_ids, clip_paths, tags, _ = pickle.load(gzip.open(os.path.join(base_path, "TuneAnnotations.pkl.gz"), "rb"))
specs = []
start = time.time()
print("Calculating spectrograms....")
for index, path in enumerate(clip_paths[:n]):
    if index in problematic:
        specs.append(np.zeros((128, 628), dtype=_type))
        continue
    signal, rate = librosa.load(os.path.join(base_path, path), sr=sr)
    specs.append(log(np.clip(librosa.feature.melspectrogram(signal, rate), epsilon, np.inf)).astype(_type))
    if (index + 1) % print_every == 0:
        print("{} spectrograms calculated".format(index + 1))
end = time.time()
specs = np.array(specs)
print("Dumping spectrograms....")
dstart = time.time()
if not use_numpy:
    pickle.dump(specs, gzip.open(os.path.join(base_path, "Spectrograms.data"), "wb") if compress else open(
        os.path.join(base_path, "Spectrograms.data"), "wb"))
else:
    if compress:
        np.savez_compressed(open(os.path.join(base_path, "Spectrograms.data"), "wb"), specs)
    else:
        np.save(open(os.path.join(base_path, "Spectrograms.data"), "wb"), specs)
dend = time.time()
logging.info("\nRESULTS:")
logging.info(
    "Spectrogram shape: {}\nCalculation time taken: {}\nDumping time taken: {}".format((specs.shape[1], specs.shape[2]),
                                                                                       end - start, dend - dstart))
print("Spectrograms dumped")
print("Maximum log amplitude: {}".format(specs.max()))
print("Minimum log amplitude: {}".format(specs.min()))
print("Spectrogram shape: {}".format((specs.shape[1], specs.shape[2])))
print("Calculation time taken: {}".format(end - start))
print("Dumping time taken: {}".format(dend - dstart))
