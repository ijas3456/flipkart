from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the Flipkart model (dummy in this case)
model = pickle.load(open('flipkart_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        # Extract input fields from form
        category = int(data.get('category', 0))
        price = float(data.get('price', 0))
        rating = float(data.get('rating', 0))

        # Combine into a NumPy array
        features = np.array([[category, price, rating]])

        # Make a prediction
        prediction = model.predict(features)
        result = prediction[0]

        return render_template('index.html', prediction_text=f'Predicted Output: {result}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
