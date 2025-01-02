from datetime import timedelta

import cv2
import numpy as np
from flask import Flask, request, jsonify

from processor.BarcodeDetector import BarcodeDetector
from processor.MeterDetector import MeterDetector
from processor.MeterMatrixDetector import MeterMatrixDetector

app = Flask(__name__)

DETECTION_METER_URL = "/v1/meter"
DETECTION_BARCODE_URL = "/v1/barcode"
DETECTION_MATRIX_URL = "/v1/matrix"

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route(DETECTION_METER_URL, methods=["POST"])
def meter_predict():
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        nparr = np.fromstring(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = MeterDetector()
        image_info = detector.detect(img)
        return jsonify({'status': 1,
                        'image_info': image_info})
    return jsonify({'status': 0})
@app.route(DETECTION_BARCODE_URL, methods=["POST"])
def barcode_predict():
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        nparr = np.fromstring(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = BarcodeDetector()
        barcode = detector.detect(img)
        return jsonify({'status': 1,
                        'barcode': barcode})
    return jsonify({'status': 0})

@app.route(DETECTION_MATRIX_URL, methods=["POST"])
def matrix_predict():
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        nparr = np.fromstring(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = MeterMatrixDetector()
        image= detector.detect(img)
        rows,cols = detector.shape(image)
        return jsonify({'status': 1,
                        'rows': rows,
                        'cols':cols})
    return jsonify({'status': 0})
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)