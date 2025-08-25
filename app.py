from flask import Flask,request,jsonify,render_template
import sklearn
import joblib
import pandas as pd
from predicting_species import predict_species 




# Create the FLASK app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict',methods = ['POST'])
def predict():
    model_name = request.form['model_name']
    sl = float(request.form['sl'])
    sw = float(request.form['sw'])
    pl = float(request.form['pl'])
    pw = float(request.form['pw'])
    output = predict_species(model_name,sl,sw,pl,pw)
    return render_template('index.html' , prediction_text=f'Prediction:{output}')

# Run the app
if __name__ == "__main__":
    app.run(debug=False)
