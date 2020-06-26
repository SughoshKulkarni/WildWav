from subprocess import run, PIPE
from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.secret_key = "VaishSughosh%$1234"
model = pickle.load(open('Bird Predictor_neural.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    bird_path = ''
    if Path('tmp/audio.wav').is_file():
        Path('tmp/audio.wav').unlink()
        return render_template('index.html', bird = bird_path)
    return render_template('index.html', bird = bird_path)

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

def allowed_file(f):
    return '.' in f and \
           f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/audio', methods=['POST'])
def audio():
    with open('./tmp/audio.wav', 'wb') as f:
        f.write(request.data)
    proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', './tmp/audio.wav'], text=True, stderr=PIPE)
    return proc.stderr

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("File too large", "error")
    return redirect(url_for("index"),413)

@app.errorhandler(500)
def request_internal_server_error(error):
    flash("Recording is too small. Please use another audio", "error")
    return redirect(url_for("index"),500)

@app.errorhandler(503)
def request_file_too_big(error):
    flash("Audio file is too large. Please use a file with size less than 5 MB", "error")
    return redirect(url_for("index"),503)

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    f = (request.files['file'])
    path = r"tmp/audio.wav"
    mfcc_=[]
    labels=['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13']
    
    if f.filename == '' or Path('tmp/audio.wav').is_file():
        if Path('tmp/audio.wav').is_file():
            X, sample_rate = librosa.load(path, sr = 44100, res_type='kaiser_best')
            mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0))
            data_test= pd.DataFrame.from_records(mfcc_, columns=labels)
            prediction = model.predict(data_test)
            class_probabilities = model.predict_proba(data_test)
            max_prob = np.max(class_probabilities)
            Path('tmp/audio.wav').unlink()
            if max_prob>0.7:
                output = prediction[0]
                if output == 'Cardinal':
                    bird_path = 'https://lh3.googleusercontent.com/ej9UTPrx_kcYkYgr_berGZ-Y7T2q0Emi9yMuexW_3fzslYXBwOmOn9NBiHlnZNDPQzq6-BFghf6CSVfcuu-22-tEUgKDpuu_nnm5Tq1DjyG1R3pAhGqV4RrqaCUpSONQPWZwS2X2pg=w2400'
                if output == 'Mourning Dove':
                    bird_path = 'https://lh3.googleusercontent.com/SkBJmdsjOi8H6kBvKeUB0cdbxuj2W036ABWkEt1JZ-aTXq0L0Eyuv4i7pmU6HBwuaHcG9P7kYuz45Qm0izsHquYQy2qyun1C2UmgIEhB6qN2XGY-Gr42DI6-0-3bnww48SdmmuOnTg=w2400'
                if output == 'Pigeon':
                    bird_path = 'https://lh3.googleusercontent.com/8du1iStq3vN954_k-_DTD6PxEoz6JDG5URiwVqT1SlUPw62DwevItGwZw2r-5fAX3yPEmSHBNaeCnucYgwZ923NueykVt0uxH1Kf8-RS-Mn_tgXg4tW69h8K9j3n0njU8tsaYuDtAQ=w2400'
                flash("It's a {}! ({:.2f}% probability)".format(output,max_prob*100), "info")
                return render_template('index.html', bird = bird_path)
            else:
                flash("Audio not recognized", "info")
                return redirect(url_for("index"))
        else:
            flash("No audio recorded or uploaded", "error")
            return redirect(url_for("index"))
        
    if f and allowed_file(f.filename):
        X, sample_rate = librosa.load(f, sr = 44100, res_type='kaiser_best')
        mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0))
        data_test= pd.DataFrame.from_records(mfcc_, columns=labels)
        prediction = model.predict(data_test)
        class_probabilities = model.predict_proba(data_test)
        max_prob = np.max(class_probabilities)
        if max_prob>0.7:
            output = prediction[0]
            if output == 'Cardinal':
                bird_path = 'https://lh3.googleusercontent.com/ej9UTPrx_kcYkYgr_berGZ-Y7T2q0Emi9yMuexW_3fzslYXBwOmOn9NBiHlnZNDPQzq6-BFghf6CSVfcuu-22-tEUgKDpuu_nnm5Tq1DjyG1R3pAhGqV4RrqaCUpSONQPWZwS2X2pg=w2400'
            if output == 'Mourning Dove':
                bird_path = 'https://lh3.googleusercontent.com/SkBJmdsjOi8H6kBvKeUB0cdbxuj2W036ABWkEt1JZ-aTXq0L0Eyuv4i7pmU6HBwuaHcG9P7kYuz45Qm0izsHquYQy2qyun1C2UmgIEhB6qN2XGY-Gr42DI6-0-3bnww48SdmmuOnTg=w2400'
            if output == 'Pigeon':
                bird_path = 'https://lh3.googleusercontent.com/8du1iStq3vN954_k-_DTD6PxEoz6JDG5URiwVqT1SlUPw62DwevItGwZw2r-5fAX3yPEmSHBNaeCnucYgwZ923NueykVt0uxH1Kf8-RS-Mn_tgXg4tW69h8K9j3n0njU8tsaYuDtAQ=w2400'        
            flash("It's a {}! ({:.2f}% probability)".format(output,max_prob*100), "info")
            return render_template('index.html', bird = bird_path)
        else:
            flash("Audio not recognized", "info")
            return redirect(url_for("index"))
    else:       
        flash("Use a WAV format file", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    
    app.run(debug=True)
