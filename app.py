# # import the following libraries
# # will convert the image to text string
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pytesseract

# # adds image processing capabilities
from PIL import Image
import pyttsx3
from googletrans import Translator
from io import BytesIO
import base64

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def recognizeText(img):
    # converts the image to result and saves it into result variable
    result = pytesseract.image_to_string(img)
    return result


def translateText(text, lang):
    p = Translator()
    # translates the text into german language
    k = p.translate(text, dest=lang)
    return k


app = Flask(__name__)
CORS(app)
white = ['http://localhost:5050', 'http://127.0.0.1:5500/']


@app.after_request
def add_cors_headers(response):
    r = request.referrer[:-1]
    if r in white:
        response.headers.add('Access-Control-Allow-Origin', r)
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Headers', 'Cache-Control')
        response.headers.add(
            'Access-Control-Allow-Headers', 'X-Requested-With')
        response.headers.add('Access-Control-Allow-Headers', 'Authorization')
        response.headers.add('Access-Control-Allow-Methods',
                             'GET, POST, OPTIONS, PUT, DELETE')
    return response


@app.route("/api", methods=["POST"])
def handle_api():
    data = request.get_json()
    if data:
        img2 = Image.open(BytesIO(base64.b64decode(data['data'])))
        text_tbr = recognizeText(img2)

    else:
        text_tbr = "lol"

    print(text_tbr)

    return jsonify(text_tbr)


if __name__ == '__main__':
    app.run(debug=True)
