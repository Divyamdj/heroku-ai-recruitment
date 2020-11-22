# from sklearn.externals import joblib
import numpy as np
# import pandas as pd
# from numpy import loadtxt
import joblib
from flask import Flask, jsonify, request, render_template
import pickle

model=joblib.load('finalized_model.pkl')

# scaler=joblib.load('Mango_UV_ShelfLife_scale.pkl')

# pca=joblib.load('Mango_UV_ShelfLife_pca.pkl')

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

	# X_1=X[0:288].reshape(1,-1)
	# X_scaled=scaler.transform(X_1)
	# X_pca=pca.transform(X_scaled)
	# result=model.predict(X_pca)

	# int_features=[int(x) for x in request.form.values()]
	final_features = [np.array(X)]
	output = model.predict_proba(final_features)
	# output=loaded_model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])

	# # output = {'results': int(result[0])}
	# output=int(result[0])

	# return jsonify(results=output)
	# output=round(output[0],1)
	return render_template('index.html', prediction_text='Output is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)