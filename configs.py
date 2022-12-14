from os.path import join as opj
import json

ROOT_DIR = "./"
def dataset_config():
    return dict(
        image_sample_step = 1,  # 비디오에서 이미지를 모두 다 볼 필요가 없다
    )
def OCR_config():
    return dict(
        langs = ["ko"],
        matching_algorithm = "hungarian",  # ["hungarian", "greedy"] 
        ocr_detector_name = "taejune",  # ["craft", "taejune"]
        ocr_recognition_name = "standard",  # ["standard"]
        # m1_labels = ["글라스박스안경", "신한은행", "행복온누리약국", "만수무병", "김숙희반찬포유"] , # 미션 1  샘플 레이블
        m3_labels = ['유리', '종이', '종이팩', '캔류', '페트', '플라스틱', '비닐'],  # 미션 3 실제 레이블 
        m3_labels_check = ['유리', '종이', '종이팩', '캔류', '페트', '플라스틱', '비닐', '비날', "플라스탁", "플러스틱"],  # 미션 3 is_mission3 용으로 만듬.
        pt_ratio = [1, 1, 1, 1],
        pt_resize_r = 2,  # m3일 때 리사이징 크기
        m1_param_readtext = {"text_threshold": 0.75, "canvas_size": 6000, "width_ths":0.5, "mag_ratio": 1.75, "low_text": 0.4, "link_threshold": 0.2},  # ocr hyperparameter
        m3_param_readtext = {"text_threshold": 0.75, "canvas_size": 6000, "width_ths":0.5, "mag_ratio": 1.75, "low_text": 0.4, "link_threshold": 0.2},  # ocr hyperparameter 
        ocr_save_models = opj(ROOT_DIR, "save_models"),
        ocr_distance_thres = 3
    )

def m1_config():
    return dict(
        m1_detector_img_size = 640,  # detetor image size
        m1_detector_device = "0",
        m1_detector_conf_thres = 0.25,
        m1_detector_iou_thres = 0.45,
        m1_detector_weights = opj(ROOT_DIR, "save_models/m1_human_detection_yolov5tl_epoch_005.pt"),  # detectir model path  

        m1_person_ensemble = "soft",  # ["hard", "soft"]
        m1_person_model_name = "resnet18", # ["resnet18", "resnet50", "resnet101", "resnext26ts", "gluon_resnet50_vlb", "tf_efficientnet_b0_ns", "tf_efficientnet_b1_ns", "tf_efficientnet_b2_ns", "vgg11", "tf_mobilenetv3_small_075"]
        m1_person_load_path = opj(ROOT_DIR, "save_models/m1_person_resnet18.pth"),
        m1_person_crop_H = 224,
        m1_person_crop_W = 128,
        m1_person_no_pretrained = True,
        m1_person_cls_names = ["male", "female", "kid"],
        m1_person_cls_dict = {0: "성인남성", 1: "성인여성", 2: "어린이"}
    )
def m2_config():
    return dict(
        m2_voice_weights = opj(ROOT_DIR, "save_models/mission2.h5"),  # 음성 model path  
    )
def m3_config():
    return dict(
        m3_person_color = { # 미션 3 실제 레이블
            '빨강' : [213, 0, 0],
            '주황' : [213, 127, 0],
            '노랑' : [213, 213, 0],
            '초록' : [0, 213, 0],
            '파랑' : [0, 0, 213],
            '보라' : [213, 0, 213],
            '하양' : [213, 213, 213],
            '검정' : [0, 0, 0],
            '회색' : [106, 106, 106]
        },

        m3_classification_dict = { # 미션 3 실제 레이블
            0: "04",    # 유리
            1: "01",    # 종이
            2: "02",    # 종이팩
            3: "03",    # 캔류
            4: "05",    # 패트
            5: "06",    # 플라스틱
            6: "07"     # 비닐
        },
        m3_trash_classification_weights = opj(ROOT_DIR, "save_models/classification.pth.tar"),  # 쓰레기 classification
        m3_trash_detect_weights = opj(ROOT_DIR, 'save_models/mission3_best.pt'), # = 쓰레기 detection
        m3_source = "/home/agc2022/inference_images",    #쓰레기 이미지가 있는 위치
        m3_img_size = 640,                                             #쓰레기 이미지 크기
        m3_conf_thres = 0.15,                                          #confidence threshhold
        m3_iou_thres = 0.50,                                           #IoU Threshhold
        m3_device = '1',                                               #GPU Number
        m3_view_img = False,                                           #Trash detect Model HyperParam으로 변경 x
        m3_save_txt = False,                                           #Trash detect Model HyperParam으로 변경 x
        m3_save_conf = False,                                          #Trash detect Model HyperParam으로 변경 x
        m3_nosave = False,                                             #Trash detect Model HyperParam으로 변경 x
        m3_classes = None,                                             #Trash detect Model HyperParam으로 변경 x
        m3_agnostic_nms = False,                                       #Trash detect Model HyperParam으로 변경 x
        m3_augment = False,                                            #Trash detect Model HyperParam으로 변경 x
        m3_update = False,                                             #Trash detect Model HyperParam으로 변경 x
        m3_project = "runs/detect",                                    #Trash detect Model HyperParam으로 변경 x
        m3_name = 'exp',                                               #Trash detect Model HyperParam으로 변경 x
        m3_exist_ok = False,                                           #Trash detect Model HyperParam으로 변경 x
        m3_no_trace = False,                                           #Trash detect Model HyperParam으로 변경 x
        m3_webcam = True,                                              #Webcam mode 지정 여부
        m3_saturation_ratio = 5,                                       #사람 옷 색상 구분을 위한 채도 변경 비율 [1~5] -> 1/2/3/4/5
        m3_Visual_ratio = 1.5                                          #사람 옷 색상 구분을 위한 명도 변경 비율 [1~2] -> 1/1.25/1.5/1.75/2
    )

def m3_pose_config():
    return dict(
        m3_pose_weight = opj(ROOT_DIR, "save_models/yolov7-w6-pose.pt"),
        # m3_pose_weight = "/home/data/AGC_final/yolov7-w6-pose.pt", #Pose를 분류하기 위한 모델의 가중치
        m3_pose_source = "/home/agc2022/inference_images",    #Pose 학습용 이미지가 있는 위치
        m3_pose_img_size = [960],                                           #Pose 학습용 이미지의 크기
        m3_pose_conf_thres = 0.25,                                          #confidence threshhold
        m3_pose_iou_thres = 0.45,                                           #IoU Threshhold
        m3_pose_device = '1',                                               #GPU Number
        m3_pose_view_img = False,                                           #Pose detect Model HyperParam으로 변경 x
        m3_pose_save_txt = False,                                           #Pose detect Model HyperParam으로 변경 x
        m3_pose_save_txt_tidl = False,                                      #Pose detect Model HyperParam으로 변경 x
        m3_pose_save_bin = False,                                           #Pose detect Model HyperParam으로 변경 x
        m3_pose_save_conf = False,                                          #Pose detect Model HyperParam으로 변경 x
        m3_pose_save_crop = False,                                          #Pose detect Model HyperParam으로 변경 x
        m3_pose_nosave = False,                                              #Pose detect Model HyperParam으로 변경 x
        m3_pose_classes = None,                                            #Pose detect Model HyperParam으로 변경 x
        m3_pose_agnostic_nms = False,                                       #Pose detect Model HyperParam으로 변경 x
        m3_pose_augment = False,                                            #Pose detect Model HyperParam으로 변경 x
        m3_pose_update = False,                                             #Pose detect Model HyperParam으로 변경 x
        m3_pose_project = "runs/detect",                                    #Pose detect Model HyperParam으로 변경 x
        m3_pose_name = False,                                               #Pose detect Model HyperParam으로 변경 x
        m3_pose_exist_ok = False,                                           #Pose detect Model HyperParam으로 변경 x
        m3_pose_line_thickness = 3,                                         #Pose detect Model HyperParam으로 변경 x
        m3_pose_hide_labels = False,                                        #Pose detect Model HyperParam으로 변경 x
        m3_pose_hide_conf = False,                                          #Pose detect Model HyperParam으로 변경 x
        m3_pose_webcam = False                                              #Webcam mode 지정 여부
    )
def predefined_json(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)
    return dict(
        m1_labels = d["place"]
    )
    
