# WildWav - Bird Sound Identifier

WildWav is a bird sound identification API that uses audio recording from the user (or browse a WAV file) and predict the type of bird.

<p align="center"><img src="/Images/Wildwav.png" width="300" height="500"></p>

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
* [Chroma feature](https://en.wikipedia.org/wiki/Chroma_feature)

For this model, we use MFCC feature. Librosa library from Python can extract MFCCs easily. It can obtained with only one line of code as shown below.
```Python
mfcc_=[]
# Get the path for file
path = r"tmp/audio.wav"
# Load the file
X, sample_rate = librosa.load(path, sr = 44100, res_type='kaiser_best')
# Insert MFCC features in mfcc_
mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0))
```

### A little bit about FFT, STFT, and MFCC
#### Fast Fourier Transform (FFT): 
Fourier transform basically transforms a waveform which contains time domain values to frequency domain.

A waveform is a time domain signal with time at its x-axis and amplitude at its y-axis.

<p align="center"><img src="/Images/waveform.png" width="500" height="300"></p>

Fourier transform removes the time domain and has frequency domains and amiplitude of each frequency bin is plotted below.
FFT is a snapshot of the entire audio in one single view.

<p align="center"><img src="/Images/fft.png" width="500" height="350"></p>

There is a problem with FFT. The FFT gives frequency in x-axis and amplitude in y-axis but loses the time feature. If the audio changes with respect to time, we will not be able to see it FFT. To overcome this, there is Short Time Fourier Transform (STFT)

#### Short Time Fourier Transform (STFT):
STFT essentially does FFT for smaller period of time window instead of the entire length of the song. STFT is a collection of these FFTs for each time window selected. The STFT shows the variation of frequency over time. In the below example the signal has two frequency throughout the entire length of the audio. This corresnponds to the FFT plot we saw above.

<p align="center"><img src="/Images/STFT.png" width="500" height="350"></p>

#### Mel Frequency Cepstral Coefficients (MFCC):
MFCCs are essentially expressing how humans hear in coefficients with number of coefficients ranging from 1-40. The algorithm has high resolution on low frequency and reduces resolution as the frequency increases. This means that higher the frequency, the algorithm uses [mel scale](https://en.wikipedia.org/wiki/Mel_scale) and groups the bunch of frequencies together and gives coefficients to each indicating the value of weight for that frequency bin. 



## Model buidling

For modeling we initially used [SVM](https://en.wikipedia.org/wiki/Support_vector_machine "Support vector machine"). But the model was not predicting with good accuracy.

We then moved to neural netwrork and built a [MLP Classifier](https://en.wikipedia.org/wiki/Multilayer_perceptron#Layers "Multilayer perceptron"). The neural network model used is extremely simple with just one hidden layer. Below is the architecture of the neural network.

<p align="center"><img src="/Images/neural.png" width="500" height="500"></p>

## Deployment

Deployment was done on [Heroku](https://www.heroku.com/) with the help of [Flask API](https://www.flaskapi.org/). The recorder interface was obtained using this [repo](https://github.com/danijel3/audio_gui "Audio GUI")

Model was built on Spyder IDE.
App was built using Flask.
Deployment was done on Heroku.
