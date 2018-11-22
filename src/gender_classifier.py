from keras.preprocessing.image import img_to_array
import numpy as np

from .helpers import get_image

avatar_k = 2
photo_k = 1
username_k = 1
fullname_k = 4

class GenderClassifier:
    def __init__(self, image_model, image_target, text_model = None):
        self.image_model = image_model
        self.text_model = text_model
        self.image_target = image_target

    def prepare_image(self, url):
        image = get_image(url)
        image = image.resize(self.image_target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image / 255.


    def analyse_result(self, avatar_prediction, photo_predictions):
        m_res = 0
        f_res = 0
        m_res = m_res + avatar_prediction[0][0] * avatar_k
        f_res = f_res + avatar_prediction[0][1] * avatar_k
        for prediction in photo_predictions:
            m_res = m_res + float(prediction['pred'][0][0]) * photo_k
            f_res = f_res + float(prediction['pred'][0][1]) * photo_k
        print(m_res)
        print(f_res)
        if m_res >= f_res:
            return 'male'
        else:
            return 'female'

    def predict(self, user):
        prepared_avatar = self.prepare_image(user.avatar)
        avatar_prediction = self.image_model.predict(prepared_avatar)
        photo_predictions = [{'value': photo_url, 'pred':  self.image_model.predict(self.prepare_image(photo_url))} for photo_url in user.photos ]
        return {
            'username': {
                'value': user.username,
                'man': 0,
                'woman': 0
            },
            'fullname': {
                'value': user.fullname,
                'man': 0,
                'woman': 0
            },
            'avatar': {
                'value': user.avatar,
                'man': float(avatar_prediction[0][0]),
                'woman': float(avatar_prediction[0][1])
            },
            'photos': [ { 'value': p['value'], 'man': float(p['pred'][0][0]), 'woman': float(p['pred'][0][1]) } for p in photo_predictions],
            'result': self.analyse_result(avatar_prediction, photo_predictions)
        }