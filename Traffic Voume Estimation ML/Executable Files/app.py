from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('RandomForest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    holiday = request.form['holiday']
    temp = float(request.form['temp'])
    rain = request.form['rain']
    snow = request.form['snow']
    weather = request.form['weather']
    date = request.form['date']
    hours = int(request.form['hours'])
    minutes = int(request.form['minutes'])
    seconds = int(request.form['seconds'])

    # Encode inputs
    holiday = 0 if holiday == 'None' else 1
    rain = 0 if rain == 'no' else 1
    snow = 0 if snow == 'no' else 1

    # Split date
    day, month, year = map(int, date.split('-'))

    # Encode weather manually
    weather_map = {'Clouds':1, 'Clear':2, 'Mist':3, 'Fog':4, 'Smoke':5}
    weather = weather_map.get(weather, 1)

    features = np.array([[holiday, temp, rain, snow, weather, day, month, year, hours, minutes, seconds]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    return render_template('result.html', prediction=int(prediction))

if __name__ == '__main__':
    app.run(debug=True)
