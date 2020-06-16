from subprocess import run, PIPE
from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import librosa
import numpy as np
import pandas as pd

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.secret_key = "VaishSughosh%$1234"
model = pickle.load(open('Bird Predictor.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

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


@app.route('/predict',methods=['GET', 'POST'])
def predict():
    f = (request.files['file'])
    path = r"tmp/audio.wav"
    mfcc_=[]
    labels=['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13','var14','var15']
    
    if f.filename == '':
        X, sample_rate = librosa.load(path, res_type='kaiser_fast')
        mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=15).T,axis=0))
        data_test= pd.DataFrame.from_records(mfcc_, columns=labels)
        prediction = model.predict(data_test)
        class_probabilities = model.predict_proba(data_test)
        max_prob = np.max(class_probabilities)
        if max_prob>0.8:
            output = prediction[0]
            flash("It's a {}!".format(output), "info")
            return redirect(url_for("index"))
        else:
            flash("Audio not recognized", "info")
            return redirect(url_for("index"))
        
    if f and allowed_file(f.filename):
        X, sample_rate = librosa.load(f, res_type='kaiser_fast')
        mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=15).T,axis=0))
        data_test= pd.DataFrame.from_records(mfcc_, columns=labels)
        prediction = model.predict(data_test)
        class_probabilities = model.predict_proba(data_test)
        max_prob = np.max(class_probabilities)
        if max_prob>0.8:
            output = prediction[0]
            flash("It's a {}!".format(output), "info")
            return redirect(url_for("index"))
        else:
            flash("Audio not recognized", "info")
            return redirect(url_for("index"))
    else:       
        flash("Use a wav format file", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    
    app.run(debug=True)
