import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

'''
    training data is 0~44
    test data is 45~49
'''
'''
    default magnitude
        mean : 0.43566742...
        var : 0.266653616394...
'''
mag_GAN_data = []        
ang_GAN_data = []
label_GAN = []

# max_x, max_y is fixed
max_y, max_x = 129, 144
mean = 0.43566742
std = 0.266653616
for trial in range(50):
    for num in range(10):
        name = 'jackson'
        filename = str(num)+"_"+name+"_"+str(trial)
        print(filename)
        
        sample_rate, samples = wavfile.read(filename + ".wav")
        frequencies, times, Zxx = signal.stft(samples, sample_rate)

        mag = np.abs(Zxx)
        ang = np.unwrap(np.angle(Zxx))

        shape = mag.shape
        diff = max_x - shape[1]
        
        if diff != 0:
            mag = np.append(mag, np.random.normal(mean, std, [129, diff]), axis = 1)
            ang = np.append(ang, np.random.normal(0, 0.25, [129, diff]), axis = 1)
            
        zero = np.zeros(10)
        zero[num] = 1
        
        mag_GAN_data.append(mag.tolist())
        ang_GAN_data.append(ang.tolist())
        label_GAN.append(zero.tolist())

'''
data is for jackson

trial 0 : 0 ~ 9 digit
trial 1 : 0 ~ 9 digit
...
trial 49 : 0 ~ 9 digit
'''



mag_GAN_data = np.array(mag_GAN_data)
ang_GAN_data = np.array(ang_GAN_data)
label_GAN = np.array(label_GAN)

mag_GAN_data = mag_GAN_data.reshape(500, 1, 129, 144)
ang_GAN_data = ang_GAN_data.reshape(500, 1, 129, 144)

np.save('mag_GAN_data', mag_GAN_data)
np.save('ang_GAN_data', ang_GAN_data)
np.save('label_GAN', label_GAN)
