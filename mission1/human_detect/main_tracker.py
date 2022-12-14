import glob
import os
import time
import sys
sys.path.insert(0, './yolov7')
# from detection_module import HumanDetector

import torch
import cv2
import json
import numpy as np
from run_human_detect import m1_human_detect
from ..person_classification.run_person import m1_person_classification


#### hyperparameter settings #####
GET_EVERY = 5
FRAME_RATE = 15
####### 여기 밑에는 노터치 #######
IOU_THRESHOLD = 0.45
IOU_NMS_THRESHOLD = 0.4
CONF_THRESHOLD = 0.25
detector = None


class SwoonTracker:
    def __init__(self, bbox_coord, initial_frame, swoon_conf, swoon_crop_img):
        self.bboxes = [bbox_coord]
        self.class_list = [1]
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
        self.swoon_crop_img = [swoon_crop_img]

    def append(self, bbox_coord, class_result, index, conf, swoon_crop_img):
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
        self.swoon_confidence.append(conf)
        self.swoon_crop_img.append(swoon_crop_img)

    def __str__(self):
        return "[Tracker Print] \n index_list: {} \n bboxes: {}".format(self.index_list, self.bboxes)


def analysis_tracker(tracker_list, image_list, args, m1_detector, device, m1_person, m1_person_transform):
    global detector
    image_list = np.array(image_list)
    output_idx = []
    output_coord = []
    output_class = []
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
        if final_index - start_index <= FRAME_RATE * 5: # 5초 이하로 쓰러진 상태면 기각
            continue

        swoon_index = tracker.index_list[:end_index+1] #[255, 265, 275, ...]
        swoon_class = tracker.class_list[:end_index+1] # [1, 1, 1, ...]
        swoon_confidence = tracker.swoon_confidence[:end_index+1]

        if sum(swoon_class) < 0.35 * len(swoon_class): # "쓰러짐 구간 중" 쓰러진 사람이 35% 이하일 경우 기각
            continue

        # width or height 가 너무 작은 경우 예외 처리
        init_width = tracker.bboxes[0][2]
        init_height = tracker.bboxes[0][3]
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
        
        # 여기서 부터 2022년도에 추가한것
        start = 0
        end = 0
        is_swoon = False
        for id, class_ in enumerate(tracker.class_list):
            if class_ == 1:
                if is_swoon:
                    end = id
                else:
                    start = end = id
                    is_swoon = True
            else:
                if is_swoon and end - start > FRAME_RATE * 5: # 5초 이상 쓰러진 상태면 출력
                    output_idx.append(tracker.index_list[start])
                    output_coord.append(tracker.bboxes[start])                    
                    output_class.append(m1_person_classification(m1_person, tracker.swoon_crop_img[start], m1_person_transform, ensemble=args))
                is_swoon = False
        if is_swoon:
            end = tracker.index_list[-1]
            if end - start > FRAME_RATE * 5: # 5초 이상 쓰러진 상태면 출력
                output_idx.append(tracker.index_list[start])
                output_coord.append(tracker.bboxes[start])                    
                output_class.append(m1_person_classification(m1_person, tracker.swoon_crop_img[start], m1_person_transform, ensemble=args))
            
        '''
        #Find first swoon
        prev_frame = swoon_index[0] - GET_EVERY + 1


        index_list = list(range(prev_frame, swoon_index[0]))


        def get_first_swoon(index_list): #이진탐색으로 최초 쓰러진 위치를 탐색
            pivot = (len(index_list) - 1) // 2

            out = m1_human_detect(cv2.imread(image_list[index_list[pivot]]), m1_detector, args.m1_detector_img_size, device, args.m1_detector_conf_thres, args.m1_detector_iou_thres)
            # out = detector.predict(cv2.imread(image_list[index_list[pivot]]), IOU_NMS_THRESHOLD, CONF_THRESHOLD)
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
            out = m1_human_detect(cv2.imread(image_list[index_list[pivot]]), m1_detector, args.m1_detector_img_size, device, args.m1_detector_conf_thres, args.m1_detector_iou_thres)
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
        '''


    return output_idx, output_coord, output_class

def xywh2xyxy(box):
    x,y,w,h = box
    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2
    return [x1,y1,x2,y2]

def cal_iou(boxA, boxB):
    boxA = xywh2xyxy(boxA)
    boxB = xywh2xyxy(boxB)
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


# def main():
#     global detector

#     test_folder = '/home/jinyong/yolov7/sample_dataset/img/cam1_01/images'
    
#     assert os.path.exists(test_folder)

# #    video_list = sorted(glob.glob(os.path.join(test_folder, "*")))
#     basepath = os.path.dirname(os.path.realpath(__file__))
#     ### 모델 선언 ####
#     weight_path = '/home/jinyong/yolov7/runs/train/Final/yolov5tl_epoch_005.pt' 
#     image_size = 640 
#     ### 모델 선언 #### 
#     detector = HumanDetector(weight_path, image_size)
#     OUTPUT_FILENAME = os.path.join(basepath, 'answersheet_swoogskkuedu.json')
    
#     output_json = {'answer':{}}
#  #   for idx, video in enumerate(video_list):
#    #     image_list = sorted(glob.glob(os.path.join(video, "*.*")))
#     image_list = sorted(glob.glob(os.path.join(test_folder, "*.*")))

#     tracker_list = []
#     start_video = time.time()
    
#     for idx, image in enumerate(image_list):

#         if idx == 0: continue
#         if idx % GET_EVERY != 0: continue # GET_EVERY 마다 inference 수행

#         frame = cv2.imread(image) 
#         out = detector.predict(frame, IOU_NMS_THRESHOLD, CONF_THRESHOLD)
#         output_class = out['out_class']
#         coordinates = out['label'] 
#         confidences = out['score']

        
#         swoon_coords = []
#         swoon_confidence = []
#         for id, (coord, class_, conf) in enumerate(zip(coordinates, output_class, confidences)): # 한 프레임 안에 쓰러진 사람 좌표 append
#             if class_ == 1: # swoon case
#                 swoon_coords.append(coord)
#                 swoon_confidence.append(conf)
                
#         if len(tracker_list) == 0: # 동영상 속 쓰러진 사람(들)이 처음 발견되면, tracker 생성
#             for swoon_coord, swoon_conf in zip(swoon_coords, swoon_confidence):
#                 tracker_list.append(SwoonTracker(swoon_coord, image, idx, swoon_conf))
#         else:
#             if len(swoon_coords) == 0: #tracker가 있지만, 쓰러진 사람이 탐지되지 않을 경우
#                 for i, tracker in enumerate(tracker_list):
#                     tracker_list[i].append([], 0, idx, image, 0) # 탐지되지 않은 정보를 모든 tracker에 append
#             else: # tracker 가 있고, 쓰러진 사람이 탐지될 경우
#                 swoon_matching = [False] * len(swoon_coords) # 쓰러짐 좌표가 매칭이 되면 True로 변경
#                 tracker_matching = [False] * len(tracker_list)
#                 for i, (swoon_coord, swoon_conf) in enumerate(zip(swoon_coords, swoon_confidence)):
#                     max_iou = -1
#                     max_index = -1
#                     for j, tracker in enumerate(tracker_list):
#                         # print(tracker.average_frame.tolist(), swoon_coord)
#                         get_iou = cal_iou(tracker.bboxes[0], swoon_coord)
#                         if max_iou < get_iou and not tracker_matching[j]:
#                             max_iou = get_iou
#                             max_index = j
#                     if max_iou > IOU_THRESHOLD:
#                         swoon_matching[i] = True
#                         tracker_list[max_index].append(swoon_coord, 1, idx, image, swoon_conf)
#                         tracker_matching[max_index] = True
#                 for z, track_bool in enumerate(tracker_matching):
#                     if not track_bool:
#                         tracker_list[z].append([], 0, idx, image, 0)
#                 for match_result, swoon, swoon_conf in zip(swoon_matching, swoon_coords, swoon_confidence): # 매칭되지 않은 쓰러진 좌표가 있다면, 그 좌표를 시작점으로 새로운 Tracker 생성
#                     if not match_result:
#                         tracker_list.append(SwoonTracker(swoon, image, idx, swoon_conf))

    
#     print("time to infer on one video:", time.time() - start_video)
#     final_decision, final_decision_confidence = analysis_tracker(tracker_list, image_list)
#     for img_out, tracker_out in zip(image_list, final_decision):          
#         box_dict = {'box':tracker_out}                
#         output_json['answer'][f'{img_out.split("/")[-1]}'] = box_dict

#     #print(output_json)
#     with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
#         json.dump(output_json, f)

# if __name__ == "__main__":
#     main()