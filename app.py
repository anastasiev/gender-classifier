from os.path import join

from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer

from keras.models import load_model

import json

from src.helpers import get_model, fetch_user
from src.gender_classifier import GenderClassifier

print('Instagram user gender recognition app')

test_user = 'dmytro.anastasiev'
image_model_weights = '/Users/danastasiev/Diploma/my_models/saved_models/mobile_last_dense_retrain.h5'
text_model_weights = '/Users/danastasiev/Diploma/text_recognition/frequency_order_2grams_heuristics_vowels_full_4086_to_32.h5'
img_width, img_height = 256, 256

print('Getting model...')
image_model = get_model(image_model_weights, img_width, img_height)
image_model._make_predict_function()
text_model = load_model(text_model_weights)
text_model._make_predict_function()
classifier = GenderClassifier(image_model, (img_width, img_height), text_model)


# Define a flask app
app = Flask(__name__,
            static_folder=join("website", "static"),
            template_folder=join("website", "templates"))

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        username = request.form['username']
        print('Fetching user...')
        user = fetch_user(username)
        print('Making prediction...')
        prediction = classifier.predict(user)
        return json.dumps(prediction)
    return None

if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    print('Model loaded. Check http://localhost:5000/')
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
