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
  


        dato_crudo = [ecioxx, rsrpxx, txpowerxx, blerxx, rssixx, speechcodecrxx]
        

        df_test = pd.DataFrame([dato_crudo], columns=[
            'wcdma_aset_ecio_avg',
            'wcdma_aset_rscp_avg',
            'wcdma_txagc',
            'wcdma_bler_average_percent_all_channels',
            'wcdma_rssi',
            'gsm_speechcodecrx'
        ])


        y_pred = modelopredictor.predict(df_test)


        #print(data)
       # prediction = str(model.predict(data)[0])
       # pred_class = class_dict[prediction]

        pred_class = float(y_pred[0])
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)