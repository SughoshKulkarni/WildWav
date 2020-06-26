# WildWav - Bird sound identifier

WildWav is a bird sound identification API that uses audio recording from the user (or browse a WAV file) and predict the type of bird.

---
## Data collection

Data collection for this type of problem can be difficult. Fortunately, there are several websites to collect crowdsourced audio files.

YouTube has several videos of bird songs of varying length in time. We downloaded the video and converted them to 5-6 seconds length audio files in WAV format. Totally 399 samples of three birds (Cardinal, Mourning dove, and Pigeon) were prepared with 133 audio samples for each bird.

| Bird  | Number of audio samples | Method of collection |
| :-------------: | :-------------: | :-------------: |
| Cardinal  | 133  | YouTube video to WAV conversion |
| Mourning Dove  | 133  | YouTube & manually created recordings |
| Pigeon  | 133  | YouTube video to WAV conversion |

## Data preprocessing

Audio processing is one of the most challenging tasks there exists. Audio has so many features. The challenge is to understand which feature to decide as the best feature for the model.

There are many types of features from an audio. Some are-
* [Short time fourier transform (STFT)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
* [Mel-frequency cepstrum coefficients  (MFCC)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22).)

Librosa library from Python can extract MFCCs easily.
### A little bit about FFT, STFT, and MFCC
## Model buidling
## Deployment

The modeling involves 
The data is preprocessed using to extract [MFCC features]() using librosa library in python. The features are fed to a neural network model to identify the type of bird.

Model was built on Spyder IDE.
App was built using Flask.
Deployment was done on Heroku.
