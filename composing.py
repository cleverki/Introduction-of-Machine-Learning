from scipy import signal
from scipy.io import wavfile

import numpy as np

# load data
ang_data = np.load('.\\recordings\\ang_GAN_data.npy')
mag_data = np.load('.\\learning_mag_label\\sample_??.npy')

for i in range(10):
    ang = ang_data[i][0]
    mag = mag_data[i][0]

    tmp = [[0 for i in range(144)]]

    mag = np.append(mag, tmp, axis = 0).astype('complex')

    cos = np.cos(ang, dtype='complex')
    sin = np.sin(ang, dtype='complex') * 1j

    # using inverse STFT, make new sound
    value = (cos+sin) * mag
    _, xrec = signal.istft(value, 8000)

    xrec = xrec.astype('int16')
    wavfile.write(".\\learning_mag_label\\" + str(i) + "generation.wav", 8000, xrec)


