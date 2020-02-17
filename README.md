# Introduction-of-Machine-Learning
2019 Fall : Introduction of Machine Learning Project</br>
The goal of this project is to classify and generate digit wav data.</br>

Dataset
==============
Free Spoken Digit Dataset(FSDD)</br>
https://github.com/Jakobovski/free-spoken-digit-dataset</br>

Preprocessing
---------------
I used Fourier Transform to transform the data into two 2D arrays : Amplitude, Phase.</br>
Each array has time on the x-axis.</br>

1.Digit Recognition
==================
I implemented CNN(Convolutional Neural Network) to classify which digit is.</br>
Only amplitude is used for input. And the Accuracy is about 94%.</br>

2.Digit Generation
=====================
DCGAN(Deep Convolution Generative Adversarial Nets) is implemented to generate digit sound.</br>
The model generates amplitude and phase in a 2D array. Then, inverse of Fourier transform is applied to them, making new sound.</br>
Amplitude is relatively well learned, but the phase is not. In the case of amplitude, the region essential for the voice amplitude is activated. However, it is not very sensitive like real data.

