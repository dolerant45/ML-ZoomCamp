import pickle 
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'pipeline.bin'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

app = Flask('conversion-prediction')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    y_pred = pipeline.predict_proba([customer])[0, 1]

    result = {
        "conversion_probability": float(round(y_pred, 3)),
        "conversion": bool(y_pred >= 0.5),
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)