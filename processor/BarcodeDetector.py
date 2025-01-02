import os
import uuid

import imutils
import torch
import numpy as np
from PIL import Image

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
import cv2
from random import randint
from pyzbar import pyzbar as pyzbar
import zxingcpp


class BarcodeDetector(object):
    def __init__(self):
        self.img_size = 640
        self.threshold = 0.4
        self.init_model()

    def init_model(self):
        self.weights = 'weights/barcode5s.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        half = self.device.type != 'cpu'
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        if half:
            model.float()  # to FP16
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names
        self.colors = [
            (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names
        ]

    def decodeDisplay(self, image):
        barcodeData = ''
        result = zxingcpp.read_barcode(image)
        if result.valid:
            print("Found barcode {} with value '{}' (format: {})".format(result, result.text, str(result.format)))
        else:
            print("could not read barcode")
        result = zxingcpp.read_barcode(image)
        if result != None:
            print("扫描结果==》 类别： {0} 内容： {1}".format(str(result.format), result.text))
            barcodeData = result.text
        # barcodes = pyzbar.decode(image)
        #
        # for barcode in barcodes:
        #     # 提取二维码的边界框的位置
        #     # 画出图像中条形码的边界框
        #     (x, y, w, h) = barcode.rect
        #     # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #
        #     # 提取二维码数据为字节对象，所以如果我们想在输出图像上
        #     # 画出来，就需要先将它转换成字符串
        #     barcodeData = barcode.data.decode("UTF8")
        #     barcodeType = barcode.type

        # 向终端打印条形码数据和条形码类型
        # print("扫描结果==》 类别： {0} 内容： {1}".format(barcodeType, barcodeData))
        return barcodeData

    def decodeZbarDisplay(self, image):
        barcodeData = ''
        barcodes = pyzbar.decode(image)

        for barcode in barcodes:
            # 提取二维码的边界框的位置
            # 画出图像中条形码的边界框
            # (x, y, w, h) = barcode.rect
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 提取二维码数据为字节对象，所以如果我们想在输出图像上
            # 画出来，就需要先将它转换成字符串
            barcodeData = barcode.data.decode("UTF8")
            barcodeType = barcode.type

            # 向终端打印条形码数据和条形码类型
            print("扫描结果==》 类别： {0} 内容： {1}".format(barcodeType, barcodeData))
        return barcodeData

    ## 图片旋转
    def rotateBound(self, image, angle):
        print("纠正度数{}".format(angle))
        # 获取宽高
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # 提取旋转矩阵 sin cos
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像的新边界尺寸
        nW = int((h * sin) + (w * cos))
        #     nH = int((h * cos) + (w * sin))
        nH = h

        # 调整旋转矩阵
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        img = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return img

    def getMinAreaRect(self, gray):
        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # blur and threshold the image
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cc = sorted(cnts, key=cv2.contourArea, reverse=True)
        # print("cc={}".format(cc))
        if len(cc):
            rect = cv2.minAreaRect(cc[0])
            return rect
        return None

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def predict(self, img):
        global img_y
        image_info = self.detect(img)
        return image_info

    def detect(self, im):
        barcode = ''
        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        image_info = {}
        count = 0

        for det in pred:
            if det is not None and len(det):
                # print("det=>",det)
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:

                    lbl = self.names[int(cls_id)]

                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
                    count += 1

                    img_temp = im[y1 - 5:y2 + 5, x1 - 5:x2 + 5]  # 参数含义分别是：y、y+h、x、x+w
                    barcode = self.decode(img_temp)
                    if barcode == '':
                        barcode = self.zbarDecode(img_temp)
        print("barcode=", barcode)
        return barcode

    def pre_process(self, data_path):
        file_name = os.path.split(data_path)[1].split('.')[0]
        return file_name

    def decode(self, image):
        w = image.shape[1]
        angle = 0
        thre = 35
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key = '{}'.format(uuid.uuid1())
        cv2.imwrite(r'E:\work\python\yolov5-flask\tmp\draw\{}.{}'.format(key, "jpg"), gray)
        barcode = self.decodeDisplay(gray)
        if barcode == '':
            if w < 500:
                img = cv2.resize(image, (int(image.shape[1] * 4), int(image.shape[0] * 4)), Image.BICUBIC)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                barcode = self.decodeDisplay(gray)
        if barcode == '':
            rect = self.getMinAreaRect(gray)
            if rect is not None:
                angle = rect[2]
                if angle > 45:
                    angle = 90 - angle
                elif angle < -45:
                    angle = -90 - angle
                else:
                    angle = -angle
                angleImg = self.rotateBound(gray, angle)
                while (len(barcode) == 0 and thre < 200):
                    ret, thresh = cv2.threshold(angleImg, thre, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    barcode = self.decodeDisplay(thresh)
                    thre = thre + 5
        print("barcode=", barcode)
        return barcode

    def zbarDecode(self, image):
        w = image.shape[1]
        angle = 0
        thre = 35
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcode = self.decodeZbarDisplay(gray)
        if barcode == '':
            if w < 500:
                img = cv2.resize(image, (int(image.shape[1] * 4), int(image.shape[0] * 4)), Image.BICUBIC)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                barcode = self.decodeZbarDisplay(gray)
        if barcode == '':
            rect = self.getMinAreaRect(gray)
            if rect is not None:
                angle = rect[2]
                if angle > 45:
                    angle = 90 - angle
                elif angle < -45:
                    angle = -90 - angle
                else:
                    angle = -angle
                angleImg = self.rotateBound(gray, angle)
                while (len(barcode) == 0 and thre < 200):
                    ret, thresh = cv2.threshold(angleImg, thre, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    barcode = self.decodeZbarDisplay(thresh)
                    thre = thre + 5
        print("barcode=", barcode)
        return barcode
