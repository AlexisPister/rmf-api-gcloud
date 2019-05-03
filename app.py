"""Flask App Project."""

from flask import Flask, jsonify, g
from flask_cors import CORS, cross_origin
from random import uniform

from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Activation
#import cv2
import numpy as np


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the neural network 
def load_model():
    # Path of weights
    model_weights_path = "./model_weights.h5"

    # Model initialisation : resnet50 + denser layer with one cell (output) for regression
    print('initializing model')
    resnet = ResNet50(include_top=False, pooling="avg")
    model = Sequential()
    model.add(resnet)
    model.add(Dense(1))
    
    # Loading weights
    print('loading weights')
    model.load_weights(model_weights_path)
    
    return model

model = load_model()

# Rate image
def rate_face():
    test = "./Dujardin.jpg"
    img = cv2.imread(test)
    img_proc = cv2.resize(img, (350, 350))
    
    rate = model.predict(np.array([img_proc]))
    print(rate)
    print(type(rate))
    return rate




def get_rand_rate():
    return uniform(3.1, 10)

@app.route('/face', methods = ['GET'])
@cross_origin()
def post_face():
    """Return a random rate for a face."""
    json_data = {'rate': 1.7}
    res = jsonify(json_data)
    return res

@app.route('/')
@cross_origin()
def index():
    """Return homepage."""
    json_data = {}
    return jsonify(json_data)

if __name__ == '__main__':
#cache['salut'] = "salut"
    app.run(debug=True)
