from flask import Flask, render_template, request,url_for,request
import pickle
import numpy as np
import joblib


model = pickle.load(open('hp_trained_model.pkl', 'rb'))

app=Flask(__name__)


@app.route('/')
def hp_home():
    return render_template('hp_home.html')

@app.route('/hp_prediction',methods=['POST'])
def hp_predict():
    data1=request.form['sqft']
    data2=request.form['place']
    data3=request.form['yo']
    data4=request.form['tf']
    data5=request.form['bhk']
    array=np.array([[data1,data2,data3,data4,data5]])
    predict=model.predict(array)
    return render_template('hp_prediction.html',output=predict)  
          

if __name__=='__main__':
    app.run(debug=True)
