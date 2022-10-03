import pickle
from pyexpat import model
from typing import final
from flask import Flask, json,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)


model1 = pickle.load(open('model1.pkl','rb'))
model2 = pickle.load(open('model2.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1)) # Standardizing and reshaping.
    output1 = model1.predict(final_input)[0]
    output2 = model2.predict(final_input)[0]
    print(output1,output2)
    return render_template("home.html", prediction_text = 'The predicted d-current and q-current at time k+1 is {0} and {1} respectively.'.format(output1, output2))


if __name__=="__main__": # Needed to run the app.
    app.run(debug=True)