import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import librosa
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('Bird Predictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='The bird is {}'.format(output))


@app.route('/prediction',methods=['GET', 'POST'])
def prediction():
    f = (request.files['file'])
    mfcc_=[]
    labels=['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13','var14','var15']
    X, sample_rate = librosa.load(f, res_type='kaiser_fast')
    mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=15).T,axis=0))
    data_test= pd.DataFrame.from_records(mfcc_, columns=labels)
    prediction = model.predict(data_test)
    output = prediction[0]
    return render_template('index.html', prediction_text='The bird is {}'.format(output))
    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
