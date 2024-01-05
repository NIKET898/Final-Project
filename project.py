import pickle
from urllib import request

from sklearn import pipeline
from sklearn.pipeline import make_pipeline as pipe
import pandas as pd
from flask import Flask, render_template, request
from flask_cors import cross_origin

app = Flask(__name__)

model = pickle.load(open("model.pkl", 'rb'))
abc = pd.read_csv("model_data.csv")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/Contactus.html')
def contactus():
    return render_template('Contactus.html')


@app.route('/predict.html')
def mack():
    # zero = sorted(abc[''].unique())
    one = sorted(abc['City'].astype(str).unique())
    two = sorted(abc['Location'].astype(str).unique())
    three = sorted(abc['Area'].unique())
    four = sorted(abc['bhk'].unique())
    return render_template('predict.html', one=one, two=two, three=three, four=four)


@app.route('/Predict', methods=['POST'])
@cross_origin()
def predict():
    City = str(request.form.get('City'))
    Location = request.form.get('Location')
    Area = float(request.form.get('Area'))
    bhk = int(request.form.get('bhk'))

    print(City, Location, Area, bhk)
    do = pipe.predict(pd.DataFrame([[City, Location, Area, bhk]], columns=['City', 'Location', 'Area', 'bhk']))

    prediction = pipeline.predict(do)[0]

    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True)
