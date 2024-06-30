from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        IQ = float(request.form['IQ'])
        CGPA = float(request.form['CGPA'])
        features = np.array([[CGPA, IQ]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        if prediction == 1:
            prediction = 'He Got Placement'
        else:
            prediction = 'He Did Not Get Placement'
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
