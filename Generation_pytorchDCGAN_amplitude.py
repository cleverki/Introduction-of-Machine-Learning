import torch
import torch.nn as nn

import numpy as np
from torch.optim import Adam

from scipy.io import wavfile
from scipy import signal

import matplotlib.pyplot as plt


img_size = 129*144
learning_rate = 0.001
num_epoch = 50
batch_size = 10
total_trial = 50
num_channel = 1
num_seed = 100000
n_noise = 100

torch.manual_seed(num_seed)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, 4, 2, 1, bias = False),
            nn.LeakyReLU(inplace = True)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 50, 4, 2, 1, bias = False),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(inplace = True)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 70, 4, 2, 1, bias = False),
            nn.BatchNorm2d(70),
            nn.LeakyReLU(inplace = True)
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(70, 100, 4, 2, 1, bias = False),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(inplace = True)
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(100, 1, (8,9), 1, 0, bias = False),
            nn.Sigmoid()
            )

        self.cond_layer = nn.Sequential(
            nn.ConvTranspose2d(10, 10, (64, 72), bias = False),
            nn.LeakyReLU(inplace = True)
            )
        
    def forward(self, input, cond, sam_num):
        '''
        cond_input : 10 * 64 * 72
        layer1 :     50 * 64 * 72
        layer2 :     100 * 32 * 36
        layer3 :     200 * 16 * 18
        layer4 :     400 * 8 * 9
        layer5 :     1 * 1 * 1
        '''
        
        #make conditional conv layer
        cond = cond.reshape(sam_num, 10, 1, 1)
        cond_input = self.cond_layer(cond)
      
        input = self.layer1(input)
        input = torch.cat((input, cond_input), dim = 1)

        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)        

        return input

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(110, 1000, (8,9), 1, 0, bias = False),
            nn.BatchNorm2d(1000),
            nn.ReLU(True)
            )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(1000, 500, 4, 2, 1, bias = False),
            nn.BatchNorm2d(500),
            nn.ReLU(True)
            )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(500, 250, 4, 2, 1, bias = False),
            nn.BatchNorm2d(250),
            nn.ReLU(True)
            )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(250, 100, 4, 2, 1, bias = False),
            nn.BatchNorm2d(100),
            nn.ReLU(True)
            )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(100, 1, 4, 2, 1, bias = False),
            nn.ReLU(True)
            )

        
    def forward(self, noise, cond, sam_num):
        #give condition
        noise_cond = torch.cat((noise, cond), dim = 1)
        
        input = noise_cond.reshape(sam_num, 110, 1, 1)

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        
        return input


mag_GAN_data = np.load('recordings\\mag_GAN_data.npy')
ang_GAN_data = np.load('recordings\\ang_GAN_data.npy')
label_GAN = np.load('recordings\\label_GAN.npy')

mag_D = Discriminator().cuda()
mag_G = Generator().cuda()

mag_D_params = filter(lambda p: p.requires_grad, mag_D.parameters())
mag_G_params = filter(lambda p: p.requires_grad, mag_G.parameters())

num_params_mag_D = sum([np.prod(p.size()) for p in mag_D_params])
num_params_mag_G = sum([np.prod(p.size()) for p in mag_G_params])


overall_num_params = num_params_mag_D + num_params_mag_G
print("number of parameters : {}".format(overall_num_params))

optimizer_mag_D = Adam(mag_D.parameters(), lr = learning_rate)
optimizer_mag_G = Adam(mag_G.parameters(), lr = learning_rate)

criterion = nn.BCELoss()

mag_D.apply(weights_init)

G_losses = []
D_losses = []

'''
    make evaluation label
'''
eval_label = []
for i in range(10):
    tmp = []
    for j in range(10):
        if i == j:
            tmp.append(1)
        else:
            tmp.append(0)
    eval_label.append(tmp)

eval_label = torch.from_numpy(np.array(eval_label)).type(torch.FloatTensor).cuda()

'''
    get time and frequency sample to show PLT image
'''
sample_rate, samples = wavfile.read('.\\recordings\\9_theo_16.wav')
frequencies, times, Zxx = signal.stft(samples, sample_rate)


'''
    learning part
'''
for epoch in range(num_epoch):
    for trial in range(total_trial):
        mag_D.zero_grad()
        mag_G.zero_grad()

        idx = trial * 10
        
        mag_input = mag_GAN_data[idx:idx+10]
        mag_label = label_GAN[idx:idx+10]
        
        mag_input = torch.from_numpy(mag_input).type(torch.FloatTensor)
        mag_input = mag_input.cuda()
        
        mag_label = torch.from_numpy(mag_label).type(torch.FloatTensor)
        mag_label = mag_label.cuda()

        # generate fake data
        noise = torch.randn(batch_size, n_noise).cuda()

        fake_mag = mag_G(noise, mag_label, batch_size)
        
        if trial % 5 == 1:
            #learning Discriminator for real data
            output = mag_D(mag_input, mag_label, batch_size)
            output = output.view(-1).cuda()

            label = torch.full((batch_size,), 1).cuda()
            
            mag_errD_real = criterion(output, label)
            mag_errD_real.backward()
            
            #learning Discriminator for fake data
            output = mag_D(fake_mag.detach(), mag_label, batch_size)
            output = output.view(-1)

            label = torch.full((batch_size,), 0).cuda()

            mag_errD_fake = criterion(output, label)
            mag_errD_fake.backward()

            mag_errD = mag_errD_real + mag_errD_fake
            optimizer_mag_D.step()

            D_losses.append(mag_errD.item())
        
        # Learning Generator
        mag_G.zero_grad()
        label = torch.full((batch_size, ), 1).cuda()

        output = mag_D(fake_mag, mag_label, batch_size)
        output = output.view(-1)

        mag_errG = criterion(output, label)
        mag_errG.backward()

        optimizer_mag_G.step()

        G_losses.append(mag_errG.item())


    print(str(epoch) + " D loss : {}, G loss : {}".format(D_losses[epoch * 10], G_losses[epoch * 50]))

    '''
        make image for evaluation
    '''
    if epoch % 5 == 1:    
        with torch.no_grad():
            new_noise = torch.randn(10, n_noise).cuda()
            
            fake = mag_G(new_noise, eval_label, 10).detach()
            
            fake = fake.cpu().numpy()
            for i in range(10):
                plt.clf()
                plt.pcolormesh(times, frequencies, fake[i][0])
                plt.imshow(fake[i][0])
                plt.savefig('.\\learning_mag_label\\' + str(i) + '_sample_' + str(epoch) + '.png')
                
            np.save('.\\learning_mag_label\\' + 'sample_' + str(epoch), fake)

