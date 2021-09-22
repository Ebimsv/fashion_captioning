from flask import Flask, render_template, request
from PIL import Image
from module import bytes_img, caption_image, find_similar_images
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


@app.route('/sentence2image', methods=['POST', 'GET'])
def sentence2image():
    if request.method == 'POST' and request.form['submit'] == 'Recommend images':
        sentence = request.form['predicted_text']
        results = find_similar_images(sentence, 3)
        return render_template('word2vec.html', results=results)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
