import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

def train_index(num_idx, name_idx, trial_idx):
    return name_idx * 450 + num_idx * 45 + trial_idx

def test_index(num_idx, name_idx, trial_idx):
    return name_idx * 450 + num_idx * 45 + trial_idx

names = ['jackson', 'nicolas', 'theo', 'yweweler']

'''
    training data is 0~44
    test data is 45~49
'''
'''
    default magnitude
        mean : 0.43566742...
        var : 0.266653616394...
'''
mag_train_data = []        
ang_train_data = []
label_train = []

mag_test_data = []
ang_test_data = []
label_test = []

# max_x, max_y is fixed
max_y, max_x = 129, 144
mean = 0.43566742
std = 0.266653616
for trial in range(50):
    for num in range(10):
        for name in names:
            filename = str(num)+"_"+name+"_"+str(trial)
            #print(filename)
            
            sample_rate, samples = wavfile.read(filename + ".wav")
            frequencies, times, Zxx = signal.stft(samples, sample_rate)

            mag = np.abs(Zxx)
            ang = np.unwrap(np.angle(Zxx))

            plt.pcolormesh(times, frequencies, ang)
            plt.imshow(ang)
            plt.show()

            

            shape = mag.shape
            diff = max_x - shape[1]
            
            if diff != 0:
                mag = np.append(mag, np.random.normal(mean, std, [129, diff]), axis = 1)
                ang = np.append(ang, np.random.normal(0, 0.25, [129, diff]), axis = 1)
            zero = np.zeros(10)
            zero[num] = 1
            if trial < 45:
                mag_train_data.append(mag.tolist())
                ang_train_data.append(ang.tolist())
                label_train.append(zero.tolist())
            else:
                mag_test_data.append(mag.tolist())
                ang_test_data.append(ang.tolist())
                label_test.append(zero.tolist())

