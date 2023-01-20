###########################
import sys

# sys.path.append("/mission1/human_detect")
# sys.path.append("/home/agc2022/src/mission1/person_classification")
# sys.path.append("/home/agc2022/src/OCR")
# sys.path.append("/home/agc2022/src/mission3")

sys.path.append("./mission1/human_detect")
sys.path.append("./mission1/person_classification")
sys.path.append("./OCR")
sys.path.append("./mission3")
## ver1
import os
from os.path import join as opj
from glob import glob
import argparse
import multiprocessing as mp
import time
import random
from collections import defaultdict
import json
from urllib import request
import logging
import random

import torch
import numpy as np
import timm
import cv2
import albumentations as A 
from albumentations.pytorch import ToTensorV2

from main_utils.util import add_dict_to_argparser, get_current_time, extract_time
from main_utils.mission1 import get_place
from configs import dataset_config, OCR_config, m1_config, m2_config, m3_config, m3_pose_config, predefined_json
from OCR.ref.EasyOCR.easyocr import Reader
from OCR.run import run_OCR
from OCR.ocr_utils.ocr import put_all_text

### Mission 1
from mission1.human_detect.run_human_detect import m1_human_detect
from mission1.human_detect.models.experimental import attempt_load
from mission1.human_detect.utils.torch_utils import select_device
from mission1.person_classification.models.pc import PersonClassification
from mission1.person_classification.run_person import m1_person_classification
from mission1.human_detect.main_tracker import SwoonTracker, cal_iou, analysis_tracker


### Mission 2

### Mission 3
from mission3.mission3_utils import frames_to_TC, xywh2xyxy, Classification_Color, most_frequent, smooth, find_nearest, make_answer
from mission3.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from mission3.pose.yolov7.detect import pose_estimate
from mission3.utils.plots import plot_one_box
from mission3.utils.datasets import LoadImages, LoadStreams, letterbox
from mission3.utils.torch_utils import select_device, load_classifier, time_synchronized
from mission3.models.experimental import attempt_load2

def build_args():
    logging.warning("build_args start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=True)  # 실제 제출할 땐 False로 하기
    parser.add_argument("--n_video2image", type=int, default=-1)  # 실제 동영상은 길어서 테스트 기간에는 잘라서 넣자. -1은 다 생성.
    parser.add_argument("--use_DDP", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--is_test", default=True)
    parser.add_argument("--video_root_dir", type=str, default="./sample_dataset")
    parser.add_argument("--image_to_root_dir", type=str, default="./inference_images")

    parser.add_argument("--predict_m1", default=True)
    parser.add_argument("--predict_m2", default=False)
    parser.add_argument("--predict_m3", default=False)
    
    add_dict_to_argparser(parser, dataset_config())
    add_dict_to_argparser(parser, OCR_config())
    add_dict_to_argparser(parser, m1_config())
    add_dict_to_argparser(parser, m2_config())
    add_dict_to_argparser(parser, m3_config())
    add_dict_to_argparser(parser, m3_pose_config())
    
    add_dict_to_argparser(parser, predefined_json("./sample_dataset/pre_defined.json"))

    args = parser.parse_args()
    if args.debug:
        args.ref_num_dir = "./time_ref_numbers"
        args.debug_dir = "./debug"
        args.answer_dir = opj(args.debug_dir, "answer")
        args.answer_cnt = 0
        os.makedirs(args.debug_dir, exist_ok=True)
        os.makedirs(args.answer_dir, exist_ok=True)
    else:
        args.ref_num_dir = "./time_ref_numbers"
        args.api_url = os.environ["REST_ANSWER_URL"]
    return args
def video2image(args, vps, idx, to_root_dir, step_size):
    '''
    Args
        vps : video paths
        idx : idx for multi-process
        to_root_dir : image 저장될 root 경로
    '''
    vp = vps[idx]
    vn = vp.split("/")[-1][:-4]
    to_dir = opj(to_root_dir, vn)
    os.makedirs(to_dir, exist_ok=True)
    video = cv2.VideoCapture(vp)
    cnt = 0
    while True:
        ret, cap = video.read()
        if cnt == args.n_video2image: break
        if not ret: break
        if cnt % step_size == 0:
            to_path = opj(to_dir, f"{cnt:06d}.png")
            cv2.imwrite(to_path, cap)
        cnt += 1

def main_worker(args):
    logging.warning("main_worker start")
    #### model initialization ####
    #### OCR
    OCR_reader = Reader(lang_list=args.langs, gpu=True, model_storage_directory=args.ocr_save_models, user_network_directory=args.ocr_save_models, detect_network=args.ocr_detector_name, recog_network=args.ocr_recognition_name, download_enabled=False)
    OCR_time_reader = Reader(lang_list=args.langs, gpu=True, model_storage_directory=args.ocr_save_models, user_network_directory=args.ocr_save_models, detect_network="craft", recog_network=args.ocr_recognition_name, download_enabled=False)
    logging.warning(f"OCR is loaded from {opj(args.ocr_save_models, args.ocr_detector_name)}, {opj(args.ocr_save_models, args.ocr_recognition_name)}")
    
    #### mission 1 model initialization ####
    if args.predict_m1:
        device = select_device(args.m1_detector_device)
        m1_detector = attempt_load(args.m1_detector_weights, map_location=device)
        half = device.type != "cpu"
        if half:
            m1_detector.half()
        logging.warning(f"m1 human detector is loaded from {args.m1_detector_weights}")
        m1_detector.eval()
        
        m1_person = PersonClassification(args)
        m1_person.load(args.m1_person_load_path)
        m1_person.to_eval()
        logging.warning(f"m1 person classification is loaded from {args.m1_person_load_path}")
        m1_person_transform = A.Compose([
            A.Normalize(),
            A.Resize(height=args.m1_person_crop_H, width=args.m1_person_crop_W),
            ToTensorV2()
        ])
        
    #### mission 2 model initialization ####
    #### mission 3 model initialization ####
    if args.predict_m3:
        source, weights, imgsz = args.m3_pose_source, args.m3_pose_weight, args.m3_pose_img_size #Pose detecting Model과 관련된 Config
        device = select_device(args.m3_pose_device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        set_logging()

        classification_model = timm.create_model('rexnet_200', num_classes=7)  # 쓰레기 classification
        ckpt = torch.load(args.m3_trash_classification_weights)
        classification_model.load_state_dict(ckpt["state_dict"])
        classification_model.eval()
        classification_model.to(device)
        logging.warning(f"m3 trash classification is loaded from {args.m3_trash_classification_weights}")
        
        model = attempt_load2(weights, map_location=device)  # load FP32 model 포즈
        logging.warning(f"m3 pose estimation is loaded from {weights}")
        trash_model = attempt_load2(args.m3_trash_detect_weights, map_location=device)  # 쓰레기 detection
        logging.warning(f"m3 trash detection is loaded from {args.m3_trash_detect_weights}")

        stride = int(model.stride.max())  # Pose detect model stride
        imgsz = imgsz[0] #pose detect model image size

        #Pose detect model image size의 정합성 체크
        if isinstance(imgsz, (list,tuple)):
            assert len(imgsz) == 2; "height and width of image has to be specified"
            imgsz[0] = check_img_size(imgsz[0], s=stride)
            imgsz[1] = check_img_size(imgsz[1], s=stride)
        else:
            imgsz = check_img_size(imgsz, s=stride)  # check img_size

        names = trash_model.module.names if hasattr(trash_model, 'module') else trash_model.names  # get class names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        interp = cv2.INTER_LINEAR
        
        if half:
            model.half() 
            trash_model.half()
    vps = sorted(glob(opj(args.video_root_dir, "*.mp4")))
    logging.warning(f"there are {len(vps)} videos")
    for vp in vps:
        vn = vp.split("/")[-1][:-4]
        video = cv2.VideoCapture(vp)
        image_idx = 0
        tracker_list = []
        fps = video.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        r_pred = 1
        n_pred = int(frame_count * r_pred)
        start_time = None
        while True:
            ret, img = video.read()
            if image_idx == args.n_video2image: break
            if not ret: break
            if image_idx % args.image_sample_step != 0:
                continue

            if image_idx == 0:
                first_img = img
                is_mission3, _, ocr_results, matched_ocr_results, pers_mat = run_OCR(args=args, reader=OCR_reader, time_reader=OCR_time_reader, img=first_img)
                start_time = extract_time(first_img, ref_num_dir=args.ref_num_dir)
                logging.warning(f"cam no : {vn}, is mission3 : {is_mission3}, start time : {start_time}")
                if args.debug:
                    ocr_img = put_all_text(first_img, ocr_results)
                    c_ocr_img = put_all_text(first_img, matched_ocr_results)
                    # logging.warning(f"cam no : {vn}, is mission3 : {is_mission3}, start time : {start_time}")
                    to_dir = opj(args.debug_dir, "first_ocr_results")
                    os.makedirs(to_dir, exist_ok=True)
                    cv2.imwrite(opj(to_dir, f'{vn}_ocr_img.png'), ocr_img)
                    cv2.imwrite(opj(to_dir, f'{vn}_c_ocr_img.png'), c_ocr_img)

            if start_time is None: break
            
            
            # image_idx += 1
            # # if image_idx == 1000: break
            # if not is_mission3:
            #     answer = dict()
            #     answer["team_id"] = "dash2022"
            #     answer["secret"] = "aBmbuPgC2ATNqHf0"
            #     answer["answer_sheet"] = dict()
            #     answer["answer_sheet"]["cam_no"] = f"{vn[-2:]}"
            #     answer["answer_sheet"]["mission"] = "1"
            #     answer["answer_sheet"]["answer"] = dict()
            #     answer["answer_sheet"]["answer"]["place"] = "UNCLEAR" # "야탑분식"
            #     answer["answer_sheet"]["answer"]["source"] ="UNCLEAR" # "자전거"
            #     answer["answer_sheet"]["answer"]["event"] = "UNCLEAR" # "충돌쓰러짐"
            #     answer["answer_sheet"]["answer"]["person"] = "UNCLEAR" # "성인남자"
            #     answer["answer_sheet"]["answer"]["time"] = get_current_time(start_time, image_idx, args.image_sample_step)
            #     if args.debug:
            #         with open(opj(args.answer_dir, f"m1_{args.answer_cnt:06d}.json"), "w", encoding="utf-8") as f:
            #             json.dump(answer, f, indent="\t", ensure_ascii=False)
            #         args.answer_cnt += 1
            #     else:

            #         data = json.dumps(answer).encode("utf-8")
            #         req = request.Request(args.api_url, data=data)
            #         resp = request.urlopen(req)
                    
            #         resp_json = eval(resp.read().decode('utf-8'))
            #         if "OK" == resp_json["status"]:
            #             print("data requests successful!!")
            #         elif "ERROR" == resp_json["status"]:
            #             raise ValueError("Receive ERROR status. Please check your source code.")
            #     answer = dict()
            #     answer["team_id"] = "dash2022"
            #     answer["secret"] = "aBmbuPgC2ATNqHf0"
            #     answer["answer_sheet"] = dict()
            #     answer["answer_sheet"]["cam_no"] = f"{vn[-2:]}"
            #     answer["answer_sheet"]["mission"] = "2"
            #     answer["answer_sheet"]["answer"] = dict()
            #     answer["answer_sheet"]["answer"]["event"] = "UNCLEAR" # "야탑분식"
            #     answer["answer_sheet"]["answer"]["time_start"] = get_current_time(start_time, image_idx, args.image_sample_step)
            #     answer["answer_sheet"]["answer"]["time_end"] = "UNCLEAR" # "충돌쓰러짐"
            #     answer["answer_sheet"]["answer"]["person"] = "UNCLEAR" # "성인남자"
            #     answer["answer_sheet"]["answer"]["person_num"] = "UNCLEAR"
            #     if args.debug:
            #         with open(opj(args.answer_dir, f"m1_{args.answer_cnt:06d}.json"), "w", encoding="utf-8") as f:
            #             json.dump(answer, f, indent="\t", ensure_ascii=False)
            #         args.answer_cnt += 1
            #     else:

            #         data = json.dumps(answer).encode("utf-8")
            #         req = request.Request(args.api_url, data=data)
            #         resp = request.urlopen(req)
                    
            #         resp_json = eval(resp.read().decode('utf-8'))
            #         if "OK" == resp_json["status"]:
            #             print("data requests successful!!")
            #         elif "ERROR" == resp_json["status"]:
            #             raise ValueError("Receive ERROR status. Please check your source code.")
            # else:
            #     answer = dict()
            #     answer["team_id"] = "dash2022"
            #     answer["secret"] = "aBmbuPgC2ATNqHf0"
            #     answer["answer_sheet"] = dict()
            #     answer["answer_sheet"]["cam_no"] = f"{vn[-2:]}"
            #     answer["answer_sheet"]["mission"] = "3"
            #     answer["answer_sheet"]["answer"] = dict()
            #     answer["answer_sheet"]["answer"]["time"] = get_current_time(start_time, image_idx, args.image_sample_step)
            #     answer["answer_sheet"]["answer"]["recycle"] = "UNCLEAR" # "03"
            #     answer["answer_sheet"]["answer"]["person_color"] = "UNCLEAR" # "초록"
            #     if args.debug:
            #         with open(opj(args.answer_dir, f"m3_{args.answer_cnt:06d}.json"), "w", encoding="utf-8") as f:
            #             json.dump(answer, f, indent="\t", ensure_ascii=False)
            #         args.answer_cnt += 1
            #     else:
            #         data = json.dumps(answer).encode("utf-8")
            #         req = request.Request(args.api_url, data=data)
            #         resp = request.urlopen(req)
                    
            #         resp_json = eval(resp.read().decode('utf-8'))
            #         if "OK" == resp_json["status"]:
            #             print("data requests successful!!")
            #         elif "ERROR" == resp_json["status"]:
            #             raise ValueError("Receive ERROR status. Please check your source code.")
            # continue

            if not is_mission3:  # mission 1,2
                ########################### mission 1 ###########################
                if args.predict_m1:
                    if image_idx < n_pred:
                        detector_results = m1_human_detect(img, m1_detector, args.m1_detector_img_size, device, args.m1_detector_conf_thres, args.m1_detector_iou_thres, crop_img=True)
                        output_class = detector_results["out_class"]
                        coordinates = detector_results["label"]
                        confidences = detector_results["score"]
                        crop_imgs_lst = detector_results["cropped_images"]
                        
                        # 쓰러진 것만 추출
                        swoon_coords = [] 
                        swoon_confidence = []
                        swoon_crop_imgs = []
                        for id, (coord, class_, conf, crop_img) in enumerate(zip(coordinates, output_class, confidences, crop_imgs_lst)): # 한 프레임 안에 쓰러진 사람 좌표 append
                            if class_ == 1: # swoon case
                                swoon_coords.append(coord)
                                swoon_confidence.append(conf)
                                swoon_crop_imgs.append(crop_img)     
                        
                        if len(tracker_list) == 0: # 동영상 속 쓰러진 사람(들)이 처음 발견되면, tracker 생성
                            for swoon_coord, swoon_conf, swoon_crop_img in zip(swoon_coords, swoon_confidence, swoon_crop_imgs):
                                tracker_list.append(SwoonTracker(swoon_coord, image_idx, swoon_conf,swoon_crop_img))
                        else:
                            if len(swoon_coords) == 0: #tracker가 있지만, 쓰러진 사람이 탐지되지 않을 경우
                                for i, tracker in enumerate(tracker_list):
                                    tracker_list[i].append([], 0, image_idx, 0, None) # 탐지되지 않은 정보를 모든 tracker에 append
                            else: # tracker 가 있고, 쓰러진 사람이 탐지될 경우
                                swoon_matching = [False] * len(swoon_coords) # 쓰러짐 좌표가 매칭이 되면 True로 변경
                                tracker_matching = [False] * len(tracker_list)
                                for i, (swoon_coord, swoon_conf, swoon_crop_img) in enumerate(zip(swoon_coords, swoon_confidence, swoon_crop_imgs)):
                                    max_iou = -1
                                    max_index = -1
                                    for j, tracker in enumerate(tracker_list):
                                        get_iou = cal_iou(tracker.bboxes[0], swoon_coord)
                                        if max_iou < get_iou and not tracker_matching[j]:
                                            max_iou = get_iou
                                            max_index = j
                                    if max_iou > args.m1_detector_iou_thres:
                                        swoon_matching[i] = True
                                        tracker_list[max_index].append(swoon_coord, 1, image_idx, swoon_conf, swoon_crop_img)
                                        tracker_matching[max_index] = True
                                for z, track_bool in enumerate(tracker_matching):
                                    if not track_bool:
                                        tracker_list[z].append([], 0, image_idx, 0, None)
                                for match_result, swoon, swoon_conf in zip(swoon_matching, swoon_coords, swoon_confidence): # 매칭되지 않은 쓰러진 좌표가 있다면, 그 좌표를 시작점으로 새로운 Tracker 생성
                                    if not match_result:
                                        tracker_list.append(SwoonTracker(swoon, image_idx, swoon_conf, swoon_crop_img))         
                    else:
                        answer = dict()
                        answer["team_id"] = "dash2022"
                        answer["secret"] = "aBmbuPgC2ATNqHf0"
                        answer["answer_sheet"] = dict()
                        answer["answer_sheet"]["cam_no"] = f"{vn[-2:]}"
                        answer["answer_sheet"]["mission"] = "1"
                        answer["answer_sheet"]["answer"] = dict()
                        answer["answer_sheet"]["answer"]["place"] = "UNCLEAR"
                        answer["answer_sheet"]["answer"]["source"] = "UNCLEAR"
                        answer["answer_sheet"]["answer"]["event"] = "UNCLEAR"
                        answer["answer_sheet"]["answer"]["person"] = "UNCLEAR" # args.m1_person_cls_dict[s_class]
                        answer["answer_sheet"]["answer"]["time"] = get_current_time(start_time, image_idx, args.image_sample_step)
                        if args.debug:
                            with open(opj(args.answer_dir, f"{args.answer_cnt:06d}.json"), "w", encoding="utf-8") as f:
                                json.dump(answer, f, indent="\t", ensure_ascii=False)
                            args.answer_cnt += 1
                        else:
                            data = json.dumps(answer).encode("utf-8")
                            req = request.Request(args.api_url, data=data)
                            resp = request.urlopen(req)
                            
                            resp_json = eval(resp.read().decode('utf-8'))
                            logging.warning(f"received mseeage: {resp_json['msg']}")
                            if "OK" == resp_json["status"]:
                                print("data requests successful!!")
                            elif "ERROR" == resp_json["status"]:
                                raise ValueError("Receive ERROR status. Please check your source code.")
            else:  # mission 3
                answer = dict()
                answer["team_id"] = "dash2022"
                answer["secret"] = "aBmbuPgC2ATNqHf0"
                answer["answer_sheet"] = dict()
                answer["answer_sheet"]["cam_no"] = f"{vn[-2:]}"
                answer["answer_sheet"]["mission"] = "3"
                answer["answer_sheet"]["answer"] = dict()
                answer["answer_sheet"]["answer"]["time"] = get_current_time(start_time, image_idx, args.image_sample_step)
                answer["answer_sheet"]["answer"]["recycle"] = "UNCLEAR" # "03"
                answer["answer_sheet"]["answer"]["person_color"] = "UNCLEAR" # "초록"
                if args.debug:
                    with open(opj(args.answer_dir, f"m3_{args.answer_cnt:06d}.json"), "w", encoding="utf-8") as f:
                        json.dump(answer, f, indent="\t", ensure_ascii=False)
                    args.answer_cnt += 1
                else:
                    data = json.dumps(answer).encode("utf-8")
                    req = request.Request(args.api_url, data=data)
                    resp = request.urlopen(req)
                    
                    resp_json = eval(resp.read().decode('utf-8'))
                    if "OK" == resp_json["status"]:
                        print("data requests successful!!")
                    elif "ERROR" == resp_json["status"]:
                        raise ValueError("Receive ERROR status. Please check your source code.")
            image_idx += 1
        # 비디오별 트랙킹 결과 json 도출
        if not is_mission3 and start_time is not None:
            if args.predict_m1:
                swoon_frames, swoon_coords, swoon_classes = analysis_tracker(tracker_list, [], args, m1_detector, device, m1_person, m1_person_transform)  
                r_pred = 0.7
                n_pred = int(len(swoon_frames) * r_pred)
                for s_frame, s_coord, s_class in zip(swoon_frames, swoon_coords, swoon_classes):
                    place, place_center_coord = get_place(s_coord, matched_ocr_results)
                    answer = dict()
                    answer["team_id"] = "dash2022"
                    answer["secret"] = "aBmbuPgC2ATNqHf0"
                    answer["answer_sheet"] = dict()
                    answer["answer_sheet"]["cam_no"] = f"{vn[-2:]}"
                    answer["answer_sheet"]["mission"] = "1"
                    answer["answer_sheet"]["answer"] = dict()
                    answer["answer_sheet"]["answer"]["place"] = place
                    answer["answer_sheet"]["answer"]["source"] = "UNCLEAR"
                    answer["answer_sheet"]["answer"]["event"] = "UNCLEAR"
                    answer["answer_sheet"]["answer"]["person"] = "UNCLEAR" # args.m1_person_cls_dict[s_class]
                    answer["answer_sheet"]["answer"]["time"] = get_current_time(start_time, s_frame, args.image_sample_step)
                    if args.debug:
                        with open(opj(args.answer_dir, f"{args.answer_cnt:06d}.json"), "w", encoding="utf-8") as f:
                            json.dump(answer, f, indent="\t", ensure_ascii=False)
                        args.answer_cnt += 1
                    else:
                        data = json.dumps(answer).encode("utf-8")
                        req = request.Request(args.api_url, data=data)
                        resp = request.urlopen(req)
                        
                        resp_json = eval(resp.read().decode('utf-8'))
                        logging.warning(f"received mseeage: {resp_json['msg']}")
                        if "OK" == resp_json["status"]:
                            print("data requests successful!!")
                        elif "ERROR" == resp_json["status"]:
                            raise ValueError("Receive ERROR status. Please check your source code.")
                
        else:
            pass
        
if __name__ == "__main__":
    args = build_args()
    tic = time.time()
    main_worker(args)
    toc = time.time()
    logging.warning(f"time for main worker : {toc-tic:.2f}s")

    # end answer 만들기
    end_answer = dict()
    end_answer["team_id"] = "dash2022"
    end_answer["secret"] = "aBmbuPgC2ATNqHf0"
    end_answer["end_of_mission"] = "true"
    if args.debug:
        logging.warning("last" + opj(args.answer_dir, f"{args.answer_cnt:06d}.json"))
        with open(opj(args.answer_dir, f"{args.answer_cnt:06d}.json"), "w", encoding="utf-8") as f:
            json.dump(end_answer, f, indent="\t", ensure_ascii=False)
    else:
        data = json.dumps(end_answer).encode("utf-8")
        req = request.Request(args.api_url, data=data)
        resp = request.urlopen(req)
        
        resp_json = eval(resp.read().decode("utf-8"))
        logging.warning(f"received message: {resp_json['msg']}")
        if "OK" == resp_json["status"]:
            logging.warning("data requests successful!!")
        elif "ERROR" == resp_json["status"]:
            raise ValueError("Receive ERROR status. Please check your source code.") 
    
    
