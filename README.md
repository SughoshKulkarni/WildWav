# WildWav - Bird Sound Identifier
[Click here to go to the website](https://www.wildwav.com/ "WildWav.com")

Ever heard distinct bird sounds in the wild or around you and wondered which bird it is? WildWav is developed precisely to identify our flappy friend!

WildWav is a web application that analyzes the bird's sound using audio recording captured by the user (or browse a WAV file) to predict the type of bird.

<p align="center"><img src="https://photos.app.goo.gl/S7wFiv62fWSNKGb98" width="100%" height="100%"></p>

**Summary**:
The Bird Sound Identifier is a personal project of mine with my friend [Vaish]( https://www.linkedin.com/in/vaishrk93/ "Vaish-LinkedIn") to explore the possibilities of machine learning. Also, to bring an idea to life and make sure it is usable by everyone by deploying the model on the Heroku server.
For this project, we used Flask API to create the app and Python for modeling. Vaish did all the web designing using HTML. The app uses a recorder.js javascript file that we found [here](https://github.com/danijel3/audio_gui "Audio recorder").

**Functionality**:
The web app has an interface with two options:
1.	Either record to capture the bird's sound using the device's microphone OR 
2.	Browse file option to choose an audio file with a WAV format.

If you choose to **RECORD** and capture the bird's sound:
1.	Place your device where the bird's sound is most audible.
2.	Click the 'RECORD' button. The analyzer box should display the audio waveform, indicating that the recording has initiated.
3.	Wait for a few seconds (maximum 10-12 seconds) and click the 'STOP' button.
4.	Click the 'PREDICT' button and wait until the application processes the recording.
5.	The type of bird will be displayed along with an image.
6.	Click on the bird image for more information about the bird.
7.	Click 'REFRESH' to start over.

If you choose to **BROWSE** and upload the bird-sound file (.wav format):
1.	Click the 'BROWSE FILE' button.
2.	Select the bird-sound file (.wav) and click open.
3.	Click the 'PREDICT' button and wait till the file is uploaded.
4.	The type of bird will be displayed along with an image.
5.	Click on the bird image for more information about the bird.
6.	Click 'REFRESH' to start over.

**Design**:
In terms of the application, the app can work on both computer and android devices. The Chrome browser has the best visual effects.
The display automatically resizes to fit the device screen best.
The app currently does not work on iOS devices due to security reasons.

**Languages and Technologies used**:
* Python
* Flask API
* HTML
* JavaScript
* Heroku

You can check out the website [here](https://www.wildwav.com/ "WildWav.com"), or you can keep reading to understand more about how we built this app from getting the data to modeling and deployment process.

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

<p align="center"><img src="/Images/waveform.png" width="500" height="100%"></p>

Fourier transform removes the time domain and has frequency domains and amiplitude of each frequency bin is plotted below.
FFT is a snapshot of the entire audio in one single view.

<p align="center"><img src="/Images/fft.png" width="500" height="116%"></p>

There is a problem with FFT. The FFT gives frequency in x-axis and amplitude in y-axis but loses the time feature. If the audio changes with respect to time, we will not be able to see it FFT. To overcome this, there is Short Time Fourier Transform (STFT)

#### Short Time Fourier Transform (STFT):
STFT essentially does FFT for smaller period of time window instead of the entire length of the song. STFT is a collection of these FFTs for each time window selected. The STFT shows the variation of frequency over time. In the below example the signal has two frequency throughout the entire length of the audio. This corresnponds to the FFT plot we saw above.

<p align="center"><img src="/Images/STFT.png" width="500" height="116%"></p>

#### Mel Frequency Cepstral Coefficients (MFCC):
MFCCs are essentially expressing how humans hear in coefficients with number of coefficients ranging from 1-40. The algorithm has high resolution on low frequency and reduces resolution as the frequency increases. This means that higher the frequency, the algorithm uses [mel scale](https://en.wikipedia.org/wiki/Mel_scale) and groups the bunch of frequencies together and gives coefficients to each indicating the value of weight for that frequency bin. 



## Model buidling

For modeling we initially used [SVM](https://en.wikipedia.org/wiki/Support_vector_machine "Support vector machine"). But the model was not predicting with good accuracy.

We then moved to neural netwrork and built a [MLP Classifier](https://en.wikipedia.org/wiki/Multilayer_perceptron#Layers "Multilayer perceptron"). The neural network model used is extremely simple with just one hidden layer. Below is the architecture of the neural network.

<p align="center"><img src="/Images/neural.png" width="500" height="167%"></p>

## Deployment

Deployment was done on [Heroku](https://www.heroku.com/) with the help of [Flask API](https://www.flaskapi.org/). The recorder interface was obtained using this [repo](https://github.com/danijel3/audio_gui "Audio GUI")

## Prediction

The model is trained to detect three birds (Cardinal, Mourning Dove, and Pigeon). More birds will be added soon.
