import os
from os.path import join as opj
from glob import glob
import re

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
from soynlp.hangle import jamo_levenshtein

from .perspective_transform import perspective_transform, reverse_perspective_transform_coord
from .hungarian_algorithm import hungarian_algorithm

FONT = ImageFont.truetype("./OCR/font/gulim.ttc", 40)

def get_ff(args):
    """
    전체 이미지에서 첫번째 프레임 get
    """
    video_names = sorted(os.listdir(opj(args.data_root_dir, "sample_image")))
    first_frames = []
    for vn in video_names:
        img_paths = sorted(glob(opj(args.data_root_dir, "sample_image", vn, "*")))
        first_frames.append(cv2.imread(img_paths[0]))
    return first_frames
def check_mission3(args, ocr_results):  # 진범님 코드에서 ocr 결과 많으면 mission 1,2 아니면 mission 3
    """
    Return: 
        is_mission3: Mission 3 여부
        start_time: 동영상 시작 시간
        save_results: Rec, cam, 날짜, 시간 제외한 OCR 결과 반환
    """
    save_results = []
    is_mission3 = True
    if len(ocr_results) > 20:
        is_mission3 = False
    start_time = None
    for result in ocr_results:
        _, txt, conf = result
        txt = txt.replace(" ", "")
        if re.search('.*REC.*', txt) or re.search('.*cam[0-9]+.*', txt):
            continue
        if re.search('[0-9][0-9][0-9][0-9][\/][0-9][0-9][\/][0-9][0-9]', txt):
            continue
        if re.search('[0-9][0-9][\.\:\;\'][0-9][0-9][\.\:\;\'][0-9][0-9]', txt):
            start_time = txt[:2] + ':' + txt[3:5] + ':' + txt[6:8]
            continue
        save_results.append(result)
    return is_mission3, start_time, save_results
# def check_mission3(args, ocr_results):  # 진범님 코드
#     """
#     Return: 
#         is_mission3: Mission 3 여부
#         start_time: 동영상 시작 시간
#         save_results: Rec, cam, 날짜, 시간 제외한 OCR 결과 반환
#     """
#     save_results = []
#     is_mission3 = False
#     start_time = None
#     for result in ocr_results:
#         _, txt, conf = result
#         txt = txt.replace(" ", "")
#         if re.search('.*REC.*', txt) or re.search('.*cam[0-9]+.*', txt):
#             continue
#         if re.search('[0-9][0-9][0-9][0-9][\/][0-9][0-9][\/][0-9][0-9]', txt):
#             continue
#         if re.search('[0-9][0-9][\.\:\;\'][0-9][0-9][\.\:\;\'][0-9][0-9]', txt):
#             start_time = txt[:2] + ':' + txt[3:5] + ':' + txt[6:8]
#             continue
#         if txt in args.m3_labels_check:
#             is_mission3 = True
#         save_results.append(result)
#     return is_mission3, start_time, save_results
# def is_valid_time(time_lst):
#     if (0 <= int(time_lst[0]) <= 2) and (0 <= int(time_lst[2]) <= 5) and (0 <= int(time_lst[4]) <= 5):
#         return True
#     else:
#         return False
# def check_mission3(args, ocr_results):  # 바뀐 코드
#     save_results = []
#     is_mission3 = False
#     start_time = None
#     for result in ocr_results:
#         _, txt, conf = result
#         txt = txt.replace(" ", "")
#         if txt in args.m3_labels:
#             is_mission3 = True
#         if re.search('.*REC.*', txt) or re.search('.*cam[0-9]+.*', txt):
#             continue
#         time_lst = []
#         if 6 <= len(txt) <= 8:
#             for r in txt:
#                 if r.isdigit(): 
#                     time_lst.append(r)
    
#         if len(time_lst) == 6 and is_valid_time(time_lst):
#             print(time_lst)
#             start_time = f'{"".join(time_lst[:2])}:{"".join(time_lst[2:4])}:{"".join(time_lst[4:])}'
#         else:           
#             save_results.append(result)
#     return is_mission3, start_time, save_results
def IoU(b1, b2):
    """
    두 bbox b1, b2간의 IoU 반환
    """
    tl1, br1 = b1
    tl2, br2 = b2
    b1_area = (br1[0] - tl1[0]) * (br1[1] - tl1[1])
    b2_area = (br2[0] - tl2[0]) * (br2[1] - tl2[1])

    x1 = max(tl1[0], tl2[0])
    y1 = max(tl1[1], tl2[1])
    x2 = min(br1[0], br2[0])
    y2 = min(br1[1], br2[1])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w*h

    iou = inter / (b1_area + b2_area - inter)
    return iou
def check_IoU(ocr_results, o_y, o_x):
    margin_box = []
    for i, (((x1,y1), _, (x2,y2), _), txt, conf) in enumerate(ocr_results):
        x1 = max(0, x1-60)
        y1 = max(0, y1-70)
        x2 = min(o_x, x2+60)
        y2 = min(o_y, y2+10)
        margin_box.append([(x1,y1),(x2,y2)])

    new_margin_box = [margin_box[0]]
    new_result = [ocr_results[0]]
    i = 0
    j=1
    while j < len(ocr_results):
        # calculate IoU between margin boxes
        iou = IoU(new_margin_box[i], margin_box[j])

        # print(new_result[i][2], new_result[i+1][2], '\t', iou)
        if iou >= 0.2:
            (tl1, _, br1, _), txt1, conf1 = new_result[i]
            (tl2, _, br2, _), txt2, conf2 = ocr_results[j]
            x1 = min(tl1[0], tl2[0])
            y1 = min(tl1[1], tl2[1])
            x2 = max(br1[0], br2[0])
            y2 = max(br1[1], br2[1])
            new_result[i]=[((x1,y1), None, (x2,y2), None), txt1+txt2, (conf1+conf2)/2]
            new_margin_box[i] = [(max(0, x1-60), max(0, y1-70)), (min(o_x, x2+60), min(o_y, y2+10))]
        else:
            new_result.append(ocr_results[j])
            new_margin_box.append(margin_box[j])
            i +=1
        j += 1
    return new_result
def put_all_text(img, result):
    """
    ocr 결과 시각화
    """
    img = img.copy()
    def put_text(np_img, text, x, y, color=(0,0,0)):
        if len(np_img.shape) == 2: color = 0
        img = Image.fromarray(np_img)
        draw = ImageDraw.Draw(img)
        draw.text((x,y) ,text, font=FONT, fill=color, )
        np_img = np.array(img)
        return np_img
    for bbox, text, prob in result:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        cv2.rectangle(img, tl, br, (0,255,0), 2)
        img = put_text(img, text, tl[0], tl[1]-60, (0,255,0))
    return img
def ocr_preprocessing(img, tr=1, br=1, rr=1, lr=1, resize_r=2):
    H, W, C = img.shape
    img, pers = perspective_transform(img, tr=tr, br=br, rr=rr, lr=lr)
    img = cv2.resize(img, (W*resize_r, H*resize_r))
    return img, pers
def reverse_ocr_preprocessing(inv_pers_mat, ocr_results, pt_resize_r):
    """
    ocr 전처리 코드 역행
    """
    reverse_ocr_results = []
    for bbox, text, prob in ocr_results:
        tmp_bbox = []
        for x, y in bbox:
            scale_x, scale_y = x//pt_resize_r, y//pt_resize_r
            coord = [scale_x, scale_y, 1]
            reverse_coord = reverse_perspective_transform_coord(inv_pers_mat, coord)
            tmp_bbox.append(reverse_coord)
        reverse_ocr_results.append((tmp_bbox, text, prob))
    return reverse_ocr_results
    
def levenshtein(s1, s2):
    """
    두 단어의 형태소 거리
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
def matching_hungarian(ocr_results, labels):
    """
    labels에 있는 것만으로 ocr_results중 가장 비슷한 문자열과 비교해서 맞춤. 중복 허용 x
    """    
    n_label = len(labels)
    n_pred = len(ocr_results)
    d_mat = np.zeros((n_label, n_pred))
    for i, l in enumerate(labels):
        for j, r in enumerate(ocr_results):
            _, text, prob = r
            d = jamo_levenshtein(l, text)
            d_mat[i][j] = d
    min_pos = hungarian_algorithm(d_mat)

    new_ocr_results = []
    for label_idx, ocr_idx in min_pos:
        bbox, _, prob = ocr_results[ocr_idx]
        new_r = (bbox, labels[label_idx], prob)
        new_ocr_results.append(new_r)
    return new_ocr_results, ocr_results
def greedy_algorithm(mat, thres=100):
    result = []
    for i,  r in enumerate(mat):
        if r.min() < thres:
            result.append((i, r.argmin()))
    return result
def matching_greedy(ocr_results, labels, thres=100):
    """
    labels에 있는 것만으로 ocr_results중 가장 비슷한 문자열과 비교해서 맞춤. 중복 허용 o
    """
    n_label = len(labels)
    n_pred = len(ocr_results)
    d_mat = np.zeros((n_label, n_pred))
    for i, l in enumerate(labels):
        for j, r in enumerate(ocr_results):
            _, text, prob = r
            d = jamo_levenshtein(l, text)
            d_mat[i][j] = d
    min_pos = greedy_algorithm(d_mat, thres)
    new_ocr_results = []
    for label_idx, ocr_idx in min_pos:
        bbox, _, prob = ocr_results[ocr_idx]
        new_r = (bbox, labels[label_idx], prob)
        new_ocr_results.append(new_r)
    return new_ocr_results, ocr_results
