import requests
import re
import json

import io
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications.mobilenet import MobileNet

from .user import User


URL = 'https://instagram.com'
PHOTO_NUMBER = 6


def get_url(user):
    return "%s/%s/" % (URL, user)


def get_response_json(html):
    regexp = re.compile('window._sharedData = (.*);</script>')
    json_str = regexp.findall(html)[0]
    return json.loads(json_str)


def fetch_user(username):
    user_url = get_url(username)
    html = requests.get(user_url).text
    user = get_response_json(html)['entry_data']['ProfilePage'][0]['graphql']['user']
    fullname = user['full_name']
    avatar = user['profile_pic_url']
    recent_posts = user['edge_owner_to_timeline_media']['edges']
    photos = []
    photo_count = 0
    for post in recent_posts:
        if photo_count == PHOTO_NUMBER:
            break
        p = post['node']
        is_video = p['is_video']
        if not is_video:
            photos.append(p['thumbnail_src'])
            photo_count = photo_count + 1
    return User(username, fullname, avatar, photos)


def get_model(weights, img_width, img_height):
    mobile_model = MobileNet(input_shape=(img_width, img_height, 3), weights=None, depth_multiplier=1, dropout=1e-3,
                             include_top=False)
    model = Sequential()
    for layer in mobile_model.layers:
        model.add(layer)
    model.add(Flatten(input_shape=mobile_model.output_shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.load_weights(weights)
    return model


def get_image(url):
    response = requests.get(url, stream=True)
    image_content = response.content
    image = Image.open(io.BytesIO(image_content))
    return image
