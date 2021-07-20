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
    # 57616, 99664, 99000, 81682, 98993
    if request.method == 'POST' and request.form['submit'] == 'Submit':
        image = Image.open(request.files['image'])
        filename = image.fp.filename
        result = caption_image(image, filename)
        image = bytes_img(image)
        return render_template('results.html', result=result, image=image)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
