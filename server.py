from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import json

app = Flask(__name__)


# Load the model when the Flask app starts
model = None
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'})

        data = request.get_json(force=True)
        cl = float(data['cl'])
        age = float(data['age'])
        sib = float(data['sib'])
        parch = float(data['parch'])
        fare = float(data['fare'])
        female = int(data['female'])
        male = int(data['male'])

        # Make prediction using the model
        prediction = model.predict([[cl, age, sib, parch, fare, female, male]])
        output = int(prediction[0])

        # Print the result
        print("Prediction result:", output)
        
        

        return jsonify({'prediction': output})
        #json.dump({'prediction': output},default=str)
    #jsonify({'prediction': str(output)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Add CORS headers after each request
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)