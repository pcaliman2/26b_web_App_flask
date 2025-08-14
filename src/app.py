from flask import Flask, request, render_template, jsonify
from pickle import load
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Cargar la lista de modelos únicos


modelopredictor = joblib.load("../models/modelo_randon_forest_200_10_5_2.pkl")


@app.route("/", methods = ["GET", "POST"])
def index():

    pred_class = None
    if request.method == "POST":
        
        modelo = request.form.get("modelo", "")

        rsrpxx = float(request.form["rsrp"])
        ecioxx = float(request.form["ecio"])
        txpowerxx =  float(request.form["txpower"])
        blerxx =  float(request.form["bler"])
        rssixx =  float(request.form["rssi"])
        speechcodecrxx =  float(request.form["speechcode"])
  



        data = [[ecioxx, rsrpxx, txpowerxx, blerxx, rssixx, speechcodecrxx]]


        columnas = ['ecio', 'rsrp', 'txpower', 'bler', 'rssi', 'speechcodecr']
        dato_crudo = [ecioxx, rsrpxx, txpowerxx, blerxx, rssixx, speechcodecrxx]
#dato_crudo = [2006, 'ram', '2500', 'good', 'gas', 129761, 'clean', 'automatic', 'in', '2021-04-29T16:03:59-0400']


        df_test = pd.DataFrame([dato_crudo], columns=columnas)
        #print(df_test.head())


        y_pred = modelopredictor.predict(df_test)


        #print(data)
       # prediction = str(model.predict(data)[0])
       # pred_class = class_dict[prediction]

        pred_class = float(y_pred[0])
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)