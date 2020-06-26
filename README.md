# WildWav - Bird sound identifier

WildWav is a bird sound identification API that uses audio signal from the user (either a recording or a WAV file) to extract [MFCC features](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22).) using librosa library in python. The features are fed to a neural network model to identify the type of bird.

Model was built on Spyder IDE.
App was built using Flask.
Deployment was done on Heroku.
