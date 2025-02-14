from flask import Flask, render_template, request, jsonify
import traceback
import joblib
import numpy as np

app = Flask(__name__)

# Load the model (ensure the path is correct)
rf_model = joblib.load("C:/Users/CHETAN/Downloads/hak/rf_model.joblib")
print("Random Forest model loaded successfully!")


@app.route('/best')
def best():
    return render_template('best.html')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Replace with your actual index.html file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Log the data for debugging
        if not data:
            return jsonify({"error": "Invalid input, JSON data expected"}), 400

        features = [float(data.get(key)) for key in [
            "phLevel", "ecLevel", "ocLevel", "availP", 
            "exchK", "availS", "availZn", "availB", 
            "availFe", "availCu", "availMn"
        ]]
        prediction = rf_model.predict([features])[0]
        result = "Healthy" if prediction == 1 else "Unhealthy"
        return jsonify({"prediction": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
