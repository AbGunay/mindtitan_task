import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request
import _pickle as pickle

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def make_predict():
    #deserialization
    model = pickle.load(open("random_forest.pkl","rb"))
    #all kinds of error checking should go here
    test_data = request.get_json(force=True)
    #convert our json to a numpy array
    X_test = np.array(test_data["data"]).reshape(1,-1)
    #make a prediction
    prediction = model.predict(X_test)
    print ("data", prediction[0])
    #return the prediction
    return jsonify(predicted_label = str(prediction[0]))


if __name__ == '__main__':
    app.run(port = 9888, debug = True)
