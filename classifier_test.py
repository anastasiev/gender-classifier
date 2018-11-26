import os
from keras.models import load_model
from src.gender_classifier import GenderClassifier

from src.helpers import get_model, fetch_user

image_model_weights = '/Users/danastasiev/Diploma/my_models/saved_models/mobile_last_dense_retrain.h5'
text_model_weights = '/Users/danastasiev/Diploma/text_recognition/frequency_order_2grams_heuristics_vowels_full_4086_to_32.h5'
img_width, img_height = 256, 256

print('Getting model...')
image_model = get_model(image_model_weights, img_width, img_height)
image_model._make_predict_function()
text_model = load_model(text_model_weights)
text_model._make_predict_function()
classifier = GenderClassifier(image_model, (img_width, img_height), text_model)

man_dir = '/Users/danastasiev/Diploma/insta_data/man'
woman_dir = '/Users/danastasiev/Diploma/insta_data/woman'
data = {}
result = {}
for file in os.listdir(man_dir):
    if file != '.DS_Store':
        file = ''.join(file.split('.')[:-1])
        data[file] = 'male'
for file in os.listdir(woman_dir):
    if file != '.DS_Store':
        file = ''.join(file.split('.')[:-1])
        data[file] = 'female'

for username, gender in data.items():
    print('Fetching user '+ username + '...')
    try:
        user = fetch_user(username)
        print('Making prediction...')
        prediction = classifier.predict(user)
        is_correct = data[username] == prediction['result']
        print('Is prediction correct: ' + str(is_correct))
        result[username] = is_correct
    except Exception:
        print('Cannot fetch user ' + username)
    print('###################################')
correct_predictions_count = 0
for username, is_correct in result.items():
    if is_correct:
        correct_predictions_count = correct_predictions_count + 1
print('System accuracy: ' + str(correct_predictions_count / len(result)))