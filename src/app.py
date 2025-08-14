from flask import Flask, request, render_template, jsonify
from pickle import load
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Cargar la lista de modelos únicos
modelos = joblib.load("./modelos_unicos.pkl")
fabricantes = joblib.load("./Fabricantes_unicos.pkl")
estados = joblib.load("./State_unicos.pkl")

preprocessor = joblib.load("../models/preprocessor.pkl")
scaler = joblib.load("../models/scaler.pkl")
modelopredictor = joblib.load("../models/modelo_random_forest_Omega10.pkl")

def formatear_fecha(fecha_str):
    dt = datetime.strptime(fecha_str, "%Y-%m-%d")
    return dt.strftime("%Y-%m-%dT00:00:00-0400")


@app.route("/autocomplete/modelos")
def autocomplete_modelos():
    q = request.args.get("q", "").lower()
    sugerencias = [m for m in modelos if q in m.lower()][:10]
    return jsonify(sugerencias)

@app.route("/autocomplete/fabricantes")
def autocomplete_fabricantes():
    q = request.args.get("q", "").lower()
    sugerencias = [f for f in fabricantes if q in f.lower()][:10]
    return jsonify(sugerencias)

@app.route('/autocomplete/estados')
def autocomplete_estados():
    q = request.args.get('q', '').lower()
    sugerencias = [estado for estado in estados if q in estado.lower()]
    return jsonify(sugerencias)






@app.route("/", methods = ["GET", "POST"])
def index():

    pred_class = None
    if request.method == "POST":
        
        modelo = request.form.get("modelo", "")

        rsrpxx = float(request.form["rsrp"])
        ecioxx = float(request.form["ecio"])
        txpowerxx = request.form["txpower"]
        blerxx = request.form["bler"]
        rssixx = request.form["rssi"]
        speechcodecrxx = request.form["speechcode"]
        tituloo = request.form["Tituloo"]
        caja = request.form["Caja"]
        statee = request.form["statee"]
        fechaa = request.form["fechaa"]

       



        data = [[ecioxx, rsrpxx, txpowerxx, blerxx, rssixx, speechcodecrxx, tituloo, caja, statee, fechaa]]

        fecha_cruda = request.form.get("fechaa")
        fecha_formateada = formatear_fecha(fecha_cruda)


        columnas = ['year', 'manufacturer', 'model', 'condition', 'fuel', 'odometer', 'title_status', 'transmission', 'state', 'posting_date']
        dato_crudo = [ecioxx, rsrpxx, txpowerxx, blerxx, rssixx, speechcodecrxx]
#dato_crudo = [2006, 'ram', '2500', 'good', 'gas', 129761, 'clean', 'automatic', 'in', '2021-04-29T16:03:59-0400']


        df_test = pd.DataFrame([dato_crudo], columns=columnas)
        #print(df_test.head())

        X_test_processed = preprocessor.transform(df_test)
        X_test_scaled = scaler.transform(X_test_processed)
        y_pred = modelopredictor.predict(X_test_scaled)


        #print(data)
       # prediction = str(model.predict(data)[0])
       # pred_class = class_dict[prediction]

        pred_class=int(y_pred)
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)