from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model and normalizer
model = pickle.load(open('rf_acc_68.pkl', 'rb'))
normalizer = pickle.load(open('normalizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        final_features = normalizer.transform(final_features)

        # Make prediction
        prediction = model.predict(final_features)

        # Prepare result
        result = 'Positive for Liver Cirrhosis' if prediction[0] == 1 else 'Negative for Liver Cirrhosis'

        return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
