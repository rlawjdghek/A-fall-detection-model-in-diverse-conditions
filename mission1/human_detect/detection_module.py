import cv2

import numpy as np
import torch

print(torch.__version__)
print(torch.cuda.is_available())

#from yolov5_test.models.experimental import attempt_load
#from yolov5_test.utils.general import (non_max_suppression, scale_coords)
#from yolov5_test.utils.torch_utils import select_device
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clamp(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clamp(0, img_shape[0])
    boxes[:, 2] = boxes[:, 2].clamp(0, img_shape[1])
    boxes[:, 3] = boxes[:, 3].clamp(0, img_shape[0])
    return boxes


class HumanDetector:
    def __init__(self, weight_path, image_size):
       # self.confidence_thresh = 0.01
       # self.iou_thresh = 0.55 
        self.device = select_device('0') # 0,1,2,3 for gpu
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(weight_path, map_location=self.device)
        self.image_size = image_size # 640 or 1024 --> 이거 모델에 따라 다름.

        if self.half:
            self.model.half()

    def predict(self, image, iou, conf):
        h0, w0 = image.shape[:2]
        r =  self.image_size / max(h0, w0)  # resize image to img_size
        interp = cv2.INTER_AREA
        img = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        (h, w) = img.shape[:2]
        img, ratio, pad = letterbox(img, new_shape=(self.image_size,  self.image_size))
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, False)[0]

        output = non_max_suppression(pred, conf_thres=conf, iou_thres=iou)

#
        nms_result = output[0]
        coords = nms_result
        coords = clip_coords(coords, (1080, 1920))
#         print(coords)
        box = coords[:, :4].clone()
        confidence_score = coords[:, 4].clone()
        classes = coords[:, -1].clone()
        coords = scale_coords(img.shape[1:], box, shapes[0], shapes[1])
        coords = coords.cpu().detach().numpy()

        result_coord = [np.round(coord).astype(np.int).tolist() for coord in coords]
        confidence_score = [score.cpu().detach().numpy() for score in confidence_score]
        classes = [c.cpu().detach().numpy() for c in classes]
#         except:
#             return {'label': [], 'score': []}

        return {
            'out_class' : classes,
            'label': result_coord,  # x,y start, x,y end
            'score': confidence_score}
