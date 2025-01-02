import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
import cv2
from random import randint


class MeterMatrixDetector(object):

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.25
        self.max_frame = 160
        self.init_model()
    def init_model(self):

        self.weights = 'weights/meter5s.pt'
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

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color,
                          thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{} ID-{:.2f}'.format(cls_id, conf), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return image
    def filled_bboxes(self, image, bboxes, line_thickness=None):
        bg_img = np.zeros_like(image)
        # bg_img[:, :, 0] = 255
        # bg_img[:, :, 1] = 255
        # bg_img[:, :, 2] = 255
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (round(1.02*x1),round(1.02*y1)), (round(0.98*x2), round(0.98*y2) )
            # c1, c2 = (x1,y1), (x2,y2)
            # c1, c2 = (x1+20,y1), (x2-20,y2)
            cv2.rectangle(bg_img, c1, c2, (1,1,1),
                          thickness=-1, lineType=cv2.LINE_AA) # filled
        return bg_img
    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        count = 0

        frameHeight = im.shape[0]
        frameWidth = im.shape[1]

        bgimage = np.zeros_like(im)
        bgimage[:, :, 0] = 255
        bgimage[:, :, 1] = 255
        bgimage[:, :, 2] = 255

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
        im = self.filled_bboxes(im, pred_boxes)
        return im
    def shape(self, image):
        grid = cv2.resize(image, (128, 128), cv2.INTER_AREA)
        gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        print("gray shape", gray.shape)
        # print("gray",gray)
        rowsArray = []
        colsArray = []
        #
        for row in range(gray.shape[0]):
            # print(gray[row,:])
            list = np.split(gray[row,:], np.where(gray[row,:] == 0)[0])
            cols = []
            for data in list:
                # print(data)
                # print(data.sum())
                if data.sum() > 0:
                    cols.append(data)
            colsArray.append(len(cols))
        print("colsArray:{}".format(max(colsArray)))
        print("========================================")
        for row in range(gray.shape[1]):
            # print(gray[:,row])
            list = np.split(gray[:,row], np.where(gray[:,row] == 0)[0])
            rows = []
            for data in list:
                # print(data)
                # print(data.sum())
                if data.sum() > 0:
                    rows.append(data)
            rowsArray.append(len(rows))
        print("rowsArray:{}".format(max(rowsArray)))
        print("========================================")
        return max(rowsArray),max(colsArray)