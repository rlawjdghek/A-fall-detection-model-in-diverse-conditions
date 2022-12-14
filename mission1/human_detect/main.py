import glob
import os
import time
import json

import torch
import cv2
import numpy as np

from .run_human_detect import detect
from .utils.torch_utils import select_device
from .models.experimental import attempt_load

GET_EVERY = 5
FRAME_RATE = 15
IOU_THRESHOLD = 0.2
IOU_NMS_THRESHOLD = 0.4
CONF_THRESHOLD = 0.22
detector = None

class SwoonTracker:
    def __init__(self, bbox_coord, filename, initial_frame, swoon_conf):
        self.bboxes = [bbox_coord]
        self.class_list = [1]
        self.filenames = [filename]
        self.initial_frame = initial_frame
        self.swoon_confidence = [swoon_conf]
        self.index_list = [initial_frame]
        self.last_bbox = bbox_coord
        self.is_swoon = False
        self.end_frame = initial_frame
        self.swoon_by_end = False
        self.sum_frame = np.array(bbox_coord)
        self.average_frame = np.array(bbox_coord)
        self.swoon_count = 1

    def append(self, bbox_coord, class_result, index, filename, conf):
        if len(bbox_coord) != 0:
            self.last_bbox = bbox_coord
            self.swoon_count += 1
            self.sum_frame += np.array(bbox_coord)
            #print('sumframe', self.sum_frame)
            self.average_frame = self.sum_frame.astype(np.int32) / self.swoon_count
            #print('average frame', self.average_frame)
        self.bboxes.append(bbox_coord)
        self.class_list.append(class_result)
        self.index_list.append(index)
        self.filenames.append(filename)
        self.swoon_confidence.append(conf)

    def __str__(self):
        return "[Tracker Print] \n index_list: {} \n bboxes: {}".format(self.index_list, self.bboxes)


def analysis_tracker(tracker_list, image_list):
    global detector
    image_list = np.array(image_list)
    output_result = [[] for i in range(len(image_list))]
    output_confidence = [[] for i in range(len(image_list))]
    ELIMINATE_FRAMES = 6
    for idx, tracker in enumerate(tracker_list):

        start_index = final_index = tracker.initial_frame
        swoon_count = 0
        end_index = 0

        for id, (class_, index) in enumerate(zip(tracker.class_list, tracker.index_list)):
            if class_ == 1:
                final_index = index
                swoon_count += 1
                end_index = id
        if final_index - start_index <= FRAME_RATE * 9: # 9초 이하로 쓰러진 상태면 기각
            continue

        swoon_index = tracker.index_list[:end_index+1] #[255, 265, 275, ...]
        swoon_class = tracker.class_list[:end_index+1] # [1, 1, 1, ...]
        swoon_confidence = tracker.swoon_confidence[:end_index+1]

        if sum(swoon_class) < 0.35 * len(swoon_class): # "쓰러짐 구간 중" 쓰러진 사람이 35% 이하일 경우 기각
            continue

        # width or height 가 너무 작은 경우 예외 처리
        init_width = tracker.bboxes[0][2] - tracker.bboxes[0][0]
        init_height = tracker.bboxes[0][3] - tracker.bboxes[0][1]
        if min(init_width, init_height) < 25:
            print("0 box cut")
            continue
        prev_coordinate = tracker.bboxes[0] # 가장 처음 디텍트된 bounding box
        count = 0
      #  print(tracker.bboxes[:len(swoon_class)])
        prev_confidence = tracker.swoon_confidence[0]
        for idx_, (class_, coordinate, conf) in enumerate(zip(swoon_class, tracker.bboxes[:len(swoon_class)], swoon_confidence)): # 빈 bbox가 append 되었으면, 앞의 bbox coordinate으로 값을 채운다.
            if idx_ == 0: continue
            if class_ == 0:
                count += 1
            else:
                if count > 0:
                    diff_coordinate = (np.array(coordinate, dtype=np.float64) - np.array(prev_coordinate, dtype=np.float64))
                    diff_confidence = conf - prev_confidence
                    for ii, box in enumerate(tracker.bboxes[idx_-count:idx_]):
                        tracker.bboxes[idx_-count+ii] = (np.array(prev_coordinate, dtype=np.float64) + ((ii+1) / count) * diff_coordinate).astype(np.int32).tolist()
                        tracker.swoon_confidence[idx_-count+ii] = prev_confidence + ((ii+1) / count) * diff_confidence
                    count = 0
                prev_coordinate = coordinate
                prev_confidence = conf
       # print("revise -->",tracker.bboxes[:len(swoon_class)])

        swoon_class_ = [1] * len(swoon_class)
        tracker_list[idx].class_list[:end_index+1] = swoon_class_ # 11100111 --> 11111111

        #Find first swoon
        prev_frame = swoon_index[0] - GET_EVERY + 1


        index_list = list(range(prev_frame, swoon_index[0]))


        def get_first_swoon(index_list): #이진탐색으로 최초 쓰러진 위치를 탐색
            pivot = (len(index_list) - 1) // 2

            out = detector.predict(cv2.imread(image_list[index_list[pivot]]), IOU_NMS_THRESHOLD, CONF_THRESHOLD)
            output_class = out['out_class']
            coordinates = out['label'] # 확인 완료

            swoon_coords = []
            for coord, class_ in zip(coordinates, output_class):  # 한 프레임 안에 쓰러진 사람 좌표 append
                if class_ == 1:  # swoon case
                    swoon_coords.append(coord)
            max_iou = -1
            for swoon_coord in swoon_coords:
                iou = cal_iou(swoon_coord, tracker.bboxes[0])
                if max_iou < iou:
                    max_iou = iou
            if len(index_list) == 1:
                if max_iou < IOU_THRESHOLD:
                    return index_list[0] + 1
                else:
                    return index_list[0]
            else:
                if max_iou < IOU_THRESHOLD:
                    return get_first_swoon(index_list[pivot+1:])
                else:
                    return get_first_swoon(index_list[:pivot+1])

        def get_last_swoon(index_list): # 이진탐색으로 최초 쓰러진 위치를 탐색

            pivot = (len(index_list)-1) // 2
            out = detector.predict(cv2.imread(image_list[index_list[pivot]]), IOU_NMS_THRESHOLD, CONF_THRESHOLD)
            output_class = out['out_class']
            coordinates = out['label'] # 확인 완료

            swoon_coords = []
            for coord, class_ in zip(coordinates, output_class):  # 한 프레임 안에 쓰러진 사람 좌표 append
                if class_ == 1:  # swoon case
                    swoon_coords.append(coord)
            max_iou = -1
            match_coordinate = None
            for swoon_coord in swoon_coords:
                iou = cal_iou(swoon_coord, tracker.bboxes[0])
                if max_iou < iou:
                    max_iou = iou
            if len(index_list):
                if max_iou < IOU_THRESHOLD:
                    return index_list[0] - 1
                else:
                    return index_list[0]
            else:
                if max_iou < IOU_THRESHOLD:
                    get_last_swoon(index_list[:pivot+1])
                else:
                    get_last_swoon(index_list[pivot+1:])

        first_swoon_index = get_first_swoon(index_list) # 첫번째 쓰러진 위치를 가져옴.
        if first_swoon_index == 1:
            first_swoon_index = 0



        last_frame = swoon_index[-1]
        if last_frame > tracker.index_list[-3]: # 쓰러짐이 동영상 끝까지 지속될 경우
            tracker_list[idx].swoon_by_end = True
            last_swoon_index = len(image_list)-1
        else:
            last_next_frame = last_frame + GET_EVERY
            last_index_list = list(range(last_frame+1, last_next_frame))
            last_swoon_index = get_last_swoon(last_index_list)


        swoon_section = list(range(first_swoon_index, last_swoon_index+1))
        first_swoon_coord = tracker.bboxes[0]
        first_swoon_conf = tracker.swoon_confidence[0]
        if first_swoon_index != tracker.index_list[0]:
            out = detector.predict(cv2.imread(image_list[first_swoon_index]), IOU_NMS_THRESHOLD, CONF_THRESHOLD)
            coordinates = out['label']
            confs = out['score']
            max_iou = -1
            real_first_swoon_coord = tracker.bboxes[0]
            real_first_swoon_conf = 0
            for coordinate in coordinates:
                iou = cal_iou(coordinate, first_swoon_coord)
                if max_iou < iou:
                    max_iou = iou
                    real_first_swoon_conf = first_swoon_conf
                    real_first_swoon_coord = coordinate
        else:
            real_first_swoon_coord = first_swoon_coord
            real_first_swoon_conf = first_swoon_conf

        swoon_index_list = tracker.index_list[:end_index+1]
        swoon_boxes = tracker.bboxes[:end_index+1]
        swoon_confs = tracker.swoon_confidence[:end_index+1]
        i = 0
        ccount = 1
        if swoon_section[0] != swoon_index_list[0]:
            ccount = swoon_index_list[0] - swoon_section[0] + 1
        for idx_1, swoon_sec in enumerate(swoon_section):
            #print(swoon_sec, swoon_index_list[i], swoon_index_list[0], swoon_index_list[-1])
            if swoon_index_list[0] > swoon_sec:
                remain_count = swoon_index_list[0] - swoon_sec # 5 4 3 2 1
                diff_ = (np.array(first_swoon_coord, dtype=np.float64) - np.array(real_first_swoon_coord, dtype=np.float64))
                output_result[swoon_sec].append(np.round(np.array(real_first_swoon_coord) + (((ccount - remain_count) / ccount) * diff_)).astype(np.int32).tolist())
                diff_swoon = first_swoon_conf - real_first_swoon_conf
                output_confidence[swoon_sec].append(real_first_swoon_conf + ((ccount - remain_count) / ccount) * diff_swoon)
            elif swoon_index_list[0] <= swoon_sec < swoon_index_list[-1]:
                first_box = swoon_boxes[i]
                next_box = swoon_boxes[i+1]
                first_conf = swoon_confs[i]
                next_conf = swoon_confs[i+1]
               # print(swoon_boxes, next_box)
                diff = (np.array(next_box, dtype=np.float64) - np.array(first_box, dtype=np.float64))
                output_box = (np.round(np.array(first_box) + (((swoon_sec - swoon_index_list[i])/GET_EVERY) * diff))).astype(np.int32).tolist()
                output_result[swoon_sec].append(output_box)

                diff_conf = next_conf - first_conf
                output_confidence[swoon_sec].append(first_conf + ((swoon_sec - swoon_index_list[i]) / GET_EVERY) * diff_conf)
                #print('hell ,,',swoon_sec, swoon_index_list[i], swoon_index_list[0], swoon_index_list[-1])
                if swoon_sec == swoon_index_list[i+1]:
                    i += 1
            else:
                output_result[swoon_sec].append(swoon_boxes[-1])
                output_confidence[swoon_sec].append(swoon_confs[-1])

        if swoon_section[0] > 10: # 쓰러진 시작점이 영상의 초반 부분이 아니면 아래 작업 수행
            for idx_1, swoon_sec in enumerate(swoon_section[:ELIMINATE_FRAMES]): # 처음 몇 프레임은 버리는 프레임
                output_result[swoon_sec].pop()
                output_confidence[swoon_sec].pop()


    return output_result, output_confidence

def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


if __name__ == "__main__":

    # test_folder = "/home/data/"
  
    source = '/home/data/AGC_final/sample_image/sample_cam1_01/000267.png'
    weights = '/home/jeonghokim/AGC_final/human_detect/save_models/yolov5tl_epoch_005.pt'

    img_size = 640
    device = '0'
    conf_thres = 0.25
    iou_thres = 0.45

    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    
    if half:
        model.half()  # to FP1
    img = cv2.imread(source)
    a = detect(img, model, img_size, device, conf_thres, iou_thres, crop_img= True)
    print(a['out_class'])
    print(a['label'])
    print(a['score'])