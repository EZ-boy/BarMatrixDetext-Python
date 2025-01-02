"""Perform test request"""
import pprint

import requests

DETECTION_BARCODE_URL = "http://localhost:5000/v1/barcode"
DETECTION_MATRIX_URL = "http://localhost:5000/v1/matrix"
TEST_IMAGE = "images/IMG_20210701_103911.jpg"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_MATRIX_URL, files={"image": image_data}).json()

pprint.pprint(response)
