import torch
import numpy as np
from models.experimental import attempt_load
from processor.BarcodeDetector import BarcodeDetector
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
import cv2
from random import randint


class MeterDetector(object):

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.25
        self.max_frame = 160
        self.init_model()
        self.barCode = BarcodeDetector()
    def init_model(self):

        self.weights = 'weights/meter5s.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        half = self.device.type != 'cpu'
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        if half:
            model.float()  # to FP16
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names
        self.colors = [
            (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names
        ]
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
    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        barcode_info = []
        count = 0

        for det in pred:
            if det is not None and len(det):
                # print("det=>",det)
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *box, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(box[0]), int(box[1])
                    x2, y2 = int(box[2]), int(box[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
                    count += 1

                    img_temp = im[y1:y2, x1:x2]  # 参数含义分别是：y、y+h、x、x+w
                    key = '{}-{:02}'.format(lbl, count)
                    cv2.imwrite(r'E:\work\python\yolov5-flask\tmp\draw\{}.{}'.format(key, "jpg"), img_temp)
                    barcode = self.barCode.predict(img_temp)
                    barcode_info.append(barcode)
        return barcode_info