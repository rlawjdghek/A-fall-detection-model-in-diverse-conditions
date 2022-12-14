import torch
import numpy as np 


import cv2
import torch

from mission1.human_detect.utils.datasets import letterbox
from mission1.human_detect.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
def m1_human_detect(img0s, model, img_size, device, conf_thres, iou_thres, crop_img = False):    
    """
    Return
        out_class (list[int]) : 쓰러졌는지 아닌지, 0: 서있음, 1: 쓰러짐
        label (list[int]) : 바운딩박스 좌표
        score (list[float]) : 바운딩박스 confidence score
        cropped_images (list[numpy]) : 이 이미지내에서

    """
    # Load model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    # Padded resize
    img = letterbox(img0s, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    
    img = torch.from_numpy(img).to(device)
    img = img.half()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

    # result outcome
    classes = []
    labels = []
    confidence_scores = []
    cropped_imgs = []
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s.shape).round()
            
            # results
            for *xyxy, conf, cls in reversed(det):
                xyxy = torch.tensor(xyxy).view(1,4).view(-1).tolist()
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                cls = int(cls.detach().cpu().numpy())
                conf = float(conf.detach().cpu().numpy())
                # result process
                classes.append(cls)
                labels.append(xywh)
                confidence_scores.append(conf)

                if crop_img:
                    cropped_im = img0s.copy()
                    x,y,w,h =[int(x) for x in xywh] # x center, y center, width, height
                    cropped_img = cropped_img = cropped_im[y-h//2: y + h//2, x-w//2: x + w//2, :]                
            
                    cropped_imgs.append(cropped_img)
    return {'out_class': classes, 'label': labels, 'score': confidence_scores, 'cropped_images': cropped_imgs}

