from flask import Flask, render_template, request
from PIL import Image
from module import bytes_img, caption_image
import numpy as np


app = Flask(__name__)


# app routes
@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/infer', methods=['POST', 'GET'])
def infer():
    if request.method == 'POST' and request.form['submit'] == 'Submit':
        image = Image.open(request.files['image'])
        result = caption_image(image)
        image = bytes_img(image)
        return render_template('results.html', result=result, image=image)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
