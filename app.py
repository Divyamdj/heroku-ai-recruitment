import numpy as np
import joblib
from flask import Flask, jsonify, request, render_template
import pickle

model=joblib.load('finalized_model.pkl')

# app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# routes
@app.route('/predict',methods=['POST'])
def predict():
	int_features = request.form.getlist('Features')

	X=int_features[0].split()
	X=np.array(X).astype(float)

	final_features = [np.array(X)]
	output = model.predict_proba(final_features)
	# output=loaded_model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])

	return jsonify(acc=output[0][0], rej=output[0][1])
	# return render_template('index.html', prediction_text='Output is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)