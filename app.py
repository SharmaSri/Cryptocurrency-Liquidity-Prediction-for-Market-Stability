import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# ✅ Load model and scaler (no need to save them here)
with open(r'C:\Users\Admin\rf_model\rf_model\rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open(r'C:\Users\Admin\rf_model\rf_model\scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json  # <-- direct access without ['data']
    print("Input data:", data)

    input_array = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(input_array)
    prediction = rf_model.predict(new_data)
    return jsonify({'prediction': int(prediction[0])})



if __name__ == "__main__":
    app.run(debug=True)
