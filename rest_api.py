import json
from flask import Flask, request
from joblib import load

app = Flask(__name__)
model = load('./lgbr_cars.model')


@app.route('/predict', methods=['GET'])
def predict():
    instance = json.loads(request.data)['instance']
    prediction = model.predict([instance])[0].round(2)

    return {'prediction': prediction}


if __name__ == '__main__':
    app.run(debug=True)
