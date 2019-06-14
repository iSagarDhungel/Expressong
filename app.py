
from flask import Flask, render_template, request
from scipy.misc import imread, imresize
import numpy as np
import re
import sys
import os
import keras

sys.path.append(os.path.abspath("./model"))

from load import *
from PIL import Image


app = Flask(__name__)
global model, graph
model, graph = init()

import base64

def convertImage(imgData):
    #decode image from base64 to png file
    imgstr = re.search(r'base64,(.*)', str(imgData)).group(1)
    with open('captured_image.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

def cropImage():
    im=Image.open("captured_image.png").convert('LA')
    # print(im.size) #320, 240
    #im.getbbox() #left up right lower
    im2=im.crop((100, 40, 220, 200))
    print(im2.size)
    im2.save("captured_image.png")

@app.route('/')
def index():
    print("App started")
    # print(keras.__version__)
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    print("Analysis Started")
    imgData = request.get_data()
    # print(imgData)
    convertImage(imgData)
    cropImage()

    # readthe image
    x = imread('captured_image.png', mode='L')

    # make it the right size
    x = imresize(x, (48, 48))
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 48, 48, 1)
    # print(x)

    with graph.as_default():

        out = model.predict(x) #returns in one hot encoding
        print("(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)")
        print(out)
        response = np.argmax(out, axis=1) #return [number] from one hot encoding
        print(response)
        return str(np.asscalar(response))


if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=8080)
