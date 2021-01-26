from flask import Flask, request, render_template,jsonify
from flask_cors import CORS, cross_origin

import traceback
import requests
import pickle
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# @app.before_first_request
# def load_model_to_app():
#     headers  = {'x-api-key': '0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z' }
#     response = requests.get('https://esports-api.lolesports.com/persisted/gw/getLive?hl=pt-BR',headers  = headers )
#     app.logger.error(response.json())
    
@app.route('/')
@cross_origin()
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def predict():

    pkl_file = open('model.pkl', 'rb')
    model = pickle.load(pkl_file)

    pkl_file2 = open('model_columns.pkl', 'rb')
    model_columns = pickle.load(pkl_file2)

    if model:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = model.predict(query)

            return jsonify({'prediction': prediction.tolist(), 'query': query.to_json()})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
        
		#Prepare the feature vector for prediction
        pkl_file = open('cat', 'rb')
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        pkl_file = open('model.pkl', 'rb')
        
        model = pickle.load(pkl_file)
        prediction = model.predict(final_features)

        return render_template('result.html',prediction=prediction)

    
if __name__ == '__main__':
	app.run(debug=True)