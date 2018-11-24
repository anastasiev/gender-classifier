import requests
import re
import json
import numpy as np

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


vowels = [ord('a'),ord('e'),ord('y'),ord('u'),ord('i'),ord('o')]


def get_name_features(name):
    arr = np.zeros(52+26*26+4)

    # Iterate each character
    for ind, x in enumerate(name):
       arr[x-ord('a')] += 1
       arr[x-ord('a')+26] += ind+1
       # Vowels
       if x in vowels:
        arr[-1] += 1
    # Iterate every 2 characters
    for x in range(len(name)-1):
       ind = (name[x]-ord('a') + 2)*26 + (name[x+1]-ord('a'))
       arr[ind] += 1
    arr[-4] = name[-1] - ord('a')
    # Second Last character
    arr[-3] = name[-2] - ord('a')
    # Length of name
    arr[-2] = len(name)
    return np.array([arr])


def prepare_username(username):
    usernames = re.findall(r'[a-zA-Z]+', username.lower())
    if len(usernames) == 2:
        return usernames[0]
    return ''.join(usernames)


def prepare_fullname(fullname):
    first_name = fullname.split(' ')[0]
    if len(first_name) == 0:
        return ""
    clear_first_name = re.findall(r'[a-z]+', first_name.lower())
    return "" if len(clear_first_name) == 0 else clear_first_name[0]
