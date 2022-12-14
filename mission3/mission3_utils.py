import numpy as np
import json

#동영상의 시간을 뽑아내기 위해 만든 함수로 Frame Index를 받아 계산한다.
def frames_to_TC (frames):
    h = int(frames / 54000) 
    m = int(frames / 900) % 60 
    s = int((frames % 900)/15) 
    return ( "%02d:%02d:%02d" % ( h, m, s))        

#
def xywh2xyxy(bounding):
    bounding = np.array(bounding)
    output = np.zeros(4)
    output[0] = bounding[0] - bounding[2] / 2  # top left x
    output[1] = bounding[1] - bounding[3] / 2  # top left y
    output[2] = bounding[0] + bounding[2] / 2  # bottom right x
    output[3] = bounding[1] + bounding[3] / 2  # bottom right y
    output = np.clip(output, 0, 400000).astype(np.uint64)
    
    return output


# 영상의 BGR 값을 보고 색상을 추출해주는 함수 
# Input : args.m3_person_color, 쓰레기를 버리는 시점 이미지의 사람 허리 위치의 RGB 값
# Output : 쓰레기를 버리는 시점 이미지의 사람 허리 위치의 색상
def Classification_Color(Color, rgb):
    # BGR 입력 예시
    # [[tensor(55, dtype=torch.uint8), tensor(56, dtype=torch.uint8), tensor(48, dtype=torch.uint8)], 
    #  [tensor(68, dtype=torch.uint8), tensor(69, dtype=torch.uint8), tensor(61, dtype=torch.uint8)]]
    color_name = []

    for rgb_ in rgb :
        
        min = 767 #RGB Maximum+1
        differ_rgb = 0
        temp_color = 0
        sub = 0
        
        #BGR -> RGB
        rgb_[0], rgb_[1], rgb_[2] = rgb_[2], rgb_[1], rgb_[0]
    
        for key, value in Color.items() :

            differ_rgb = [a - b for a, b in zip(value, rgb_)]
            
            sub = np.sum(np.abs(differ_rgb))
            if min > sub :
                min = sub
                temp_color = key

            else :
                pass
    
        color_name.append(temp_color)    

    # 2Point의 예상하는 색깔이 다른 경우
    # ex) color_name = "하양"
    print("color_name", color_name)

    if color_name[0] != color_name[1] : 
        color_name[0] = "UNCLEAR"
        return color_name[0]
    
    else :
        return color_name[0]


# 영상이 쓰레기를 버리는 시점의 이미지에서 사람의 옷색깔을 추론
# Input : 쓰레기를 버리는 시점 이미지의 사람 허리 위치의 RGB 값
# Output : 쓰레기를 버리는 시점 이미지의 사람 허리 위치의 색상
def most_frequent(data):
    
    most_color = []
    for i in range(len(data)) :
        # if len(data[i]) > 10 :
            # most_color.append(max(data[i], key = data[i].count)) 
        most_color.append(data[i]) 

    #ex) ["회색", "회색"]
    # print("most_color" , most_color)
    return most_color


import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

#특정 값을 등분하여 Smooth하게 나누어주는 함수
#Input : y - 값, box_pts - 갯수 ex) [200.0]
#output : y값을 box_pts만큼 나눈 값 ex) [20   20    20    20    20    20    20    20    20    20]
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# 정답을 만들어 주는 함수
# Input : 
# Output : 

def make_answer(y_indices, cls_output, name, cur, final_color, args, answer) :
    print(f"y indices : {y_indices}")

    answer["answer_sheet"]["cam_no"] = name
    answer["answer_sheet"]["mission"] = "3"
    answer["answer_sheet"]["answer"] = dict()

    # 영상의 번호, 상황 발생 시간, 생활폐기물 종류, 버리는 주체의 인상착의 정보 (색상*)
    for i in range(len(y_indices)):
        y = y_indices[i]["y"]
        x = np.array(y_indices[i]["x"])
        c = np.array(cls_output[i]['y'])

        # 극댓값을 뽑아냄.
        if len(y) > 0:
            #Frame 단위가 달라 데이터 제데로 안나와서 time, recycle은 일단 Unclear로 표시했으며, 이때 때문에 모든 이미지에 대한 Json 생성
            yhat = smooth(y, 10)
            inform = find_peaks(yhat, height=430, distance=30)[0]
            # print("inform", inform)  [15 70]
            frame = x[inform]
            c_answer = c[inform]
            # print(frame, c_answer) [15 70] [5 5]
            # for f_, c_ in zip(frame, c_answer):
            #     ob = {}
            #     ob["time"] = frames_to_TC(f_)
            #     ob["recycle"] = args.m3_classification_dict[c_]
            #     ob["person_color"] = final_color[i] #len(y_indices) 감지된 사람+쓰레기 갯수??
            #     answer['answer_sheet']['answer'].append(ob)
            # ob = {}
            answer['answer_sheet']['answer']["time"] = "UNCLEAR"
            answer['answer_sheet']['answer']["recycle"] = "UNCLEAR"
            answer['answer_sheet']['answer']["person_color"] = final_color[i] #To Do 사람 여러명 동시 감지되는 경우에 어떻게 표기할지 확인 후 처리

        #감지한 횟수당 Json File 생성
    return answer

# 미사용
def find_nearest(key_point, bbox_points):
    best_idx = 0
    near = 9999999
    for bbox_idx in range(4):
        x, y = bbox_points[bbox_idx]
        w, h = abs(key_point[0] - x), abs(key_point[1] - y)
        distance = w + h
        if near > distance:
            best_idx = bbox_idx
            near = distance
            height, width = h, w
    return bbox_points[best_idx], int(height *1.5), int(width * 1.5)

