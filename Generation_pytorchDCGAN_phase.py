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
batch_size = 40
total_trial = 45
num_channel = 1
num_seed = 100000
n_noise = 100
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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
            nn.Conv2d(1, 50, 4, 2, 1, bias = False),
            nn.LeakyReLU(inplace = True)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(60, 100, 4, 2, 1, bias = False),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(inplace = True)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(100, 200, 4, 2, 1, bias = False),
            nn.BatchNorm2d(200),
            nn.LeakyReLU(inplace = True)
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(200, 400, 4, 2, 1, bias = False),
            nn.BatchNorm2d(400),
            nn.LeakyReLU(inplace = True)
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(400, 1, (8,9), 1, 0, bias = False),
            nn.Sigmoid()
            )

        self.cond_layer = nn.Sequential(
            nn.ConvTranspose2d(10, 10, (64, 72), bias = False),
            nn.LeakyReLU(inplace = True)
            )
        
    def forward(self, input, cond):
        '''
        cond_input : 10 * 64 * 72
        layer1 :     50 * 64 * 72
        layer2 :     100 * 32 * 36
        layer3 :     200 * 16 * 18
        layer4 :     400 * 8 * 9
        layer5 :     1 * 1 * 1
        '''
        
        #make conditional conv layer
        cond = cond.reshape(40, 10, 1, 1)
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
            nn.ConvTranspose2d(110, 800, (8,9), 1, 0, bias = False),
            nn.BatchNorm2d(800),
            nn.ReLU(True)
            )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(800, 400, 4, 2, 1, bias = False),
            nn.BatchNorm2d(400),
            nn.ReLU(True)
            )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(400, 200, 4, 2, 1, bias = False),
            nn.BatchNorm2d(200),
            nn.ReLU(True)
            )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 4, 2, 1, bias = False),
            nn.BatchNorm2d(100),
            nn.ReLU(True)
            )
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(100, 1, 4, 2, 1, bias = False),
            nn.Tanh()
            )

        
    def forward(self, noise, cond, sam_num):
        noise_cond = torch.cat((noise, cond), dim = 1)
        
        input = noise_cond.reshape(sam_num, 110, 1, 1)

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = input * np.pi
        
        return input
        
mag_GAN_data = np.load('recordings\\mag_GAN_data.npy')
ang_GAN_data = np.load('recordings\\ang_GAN_data.npy')
label_GAN = np.load('recordings\\label_GAN.npy')

ang_D = Discriminator().cuda()
ang_G = Generator().cuda()

ang_D_params = filter(lambda p: p.requires_grad, ang_D.parameters())
ang_G_params = filter(lambda p: p.requires_grad, ang_G.parameters())

num_params_ang_D = sum([np.prod(p.size()) for p in ang_D_params])
num_params_ang_G = sum([np.prod(p.size()) for p in ang_G_params])


overall_num_params = num_params_ang_D + num_params_ang_G
print("number of parameters : {}".format(overall_num_params))

optimizer_ang_D = Adam(ang_D.parameters(), lr = learning_rate)
optimizer_ang_G = Adam(ang_G.parameters(), lr = learning_rate)

criterion = nn.BCELoss()

ang_D.apply(weights_init)

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
        ang_D.zero_grad()
        ang_G.zero_grad()


        idx = trial*40

        ang_input = ang_GAN_data[idx : idx + 40]
        ang_input = torch.from_numpy(ang_input).type(torch.FloatTensor)
        ang_input = ang_input.cuda()
        
        ang_label = label_GAN[idx : idx + 40]
        ang_label = torch.from_numpy(ang_label).type(torch.FloatTensor)
        ang_label = ang_label.cuda()

        # generate fake data
        noise = torch.randn(40, n_noise).cuda()

        fake_ang = ang_G(noise, ang_label, 40)
        
        if trial % 5 == 1:
            #learning Discriminator for real data
            output = ang_D(ang_input, ang_label)
            output = output.view(-1).cuda()

            label = torch.full((40,), 1).cuda()

            ang_errD_real = criterion(output, label)
            ang_errD_real.backward()

            #learning Discriminator for fake data
            output = ang_D(fake_ang.detach(), ang_label)
            output = output.view(-1)

            label = torch.full((40,), 0).cuda()

            ang_errD_fake = criterion(output, label)
            ang_errD_fake.backward()

            ang_errD = ang_errD_real + ang_errD_fake
            optimizer_ang_D.step()

            D_losses.append(ang_errD.item())
        
        # Learning Generator
        ang_G.zero_grad()
        label = torch.full((40, ), 1).cuda()

        output = ang_D(fake_ang, ang_label)
        output = output.view(-1)

        ang_errG = criterion(output, label)
        ang_errG.backward()

        optimizer_ang_G.step()

        G_losses.append(ang_errG.item())

        
    print("D loss : {}, G loss : {}".format(D_losses[epoch * 9], G_losses[epoch * 45]))

    '''
        make image for evaluation
    '''
    with torch.no_grad():
        new_noise = torch.randn(10, n_noise).cuda()
        fake = ang_G(new_noise, eval_label, 10).detach()
        fake = fake.cpu().numpy()
        plt.clf()
        plt.pcolormesh(times, frequencies, fake[0][0])
        plt.imshow(fake[0][0])
        plt.savefig('.\\learning_ang\\' + str(0) + '_sample_' + str(epoch) + '.png')
        np.save('.\\learning_ang\\' + str(0) + '_sample_' + str(epoch), fake)
            
