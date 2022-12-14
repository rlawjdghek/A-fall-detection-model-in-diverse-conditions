import os
from os.path import join as opj

import numpy as np
import cv2

from .ocr_utils.ocr import check_mission3, matching_hungarian, ocr_preprocessing, reverse_ocr_preprocessing, matching_greedy

def run_OCR(args, reader, time_reader, img):
    """
    Returns
        img : 원본 이미지
        is_mission3 : m3 인지 아닌지
        start_time : 시작시간
        bbox : m1,m2 또는 m3에 해당하는 bbox. cam, start time은 제외함.
    """
    ocr_results = time_reader.readtext(img, **args.m1_param_readtext)
    #### TODO : check_mission3 보강
    is_mission3, start_time, ocr_results = check_mission3(args, ocr_results)
    if not is_mission3:  # mission 1, 2
        new_ocr_results, ocr_results = matching_greedy(ocr_results, args.m1_labels, args.ocr_distance_thres)
        pers_mat = None
    else:  # mission 3
        tr,br,rr,lr = args.pt_ratio
        m3_img = img.copy()
        
        m3_img, pers_mat = ocr_preprocessing(m3_img, tr=tr, br=br, rr=rr, lr=lr, resize_r=args.pt_resize_r)
        ocr_results = reader.readtext(m3_img, **args.m3_param_readtext)
        inv_pers_mat = np.linalg.inv(pers_mat)
        ocr_results = reverse_ocr_preprocessing(inv_pers_mat, ocr_results, args.pt_resize_r)

        _, _, ocr_results = check_mission3(args, ocr_results)
        if args.matching_algorithm == "hungarian":
            new_ocr_results, ocr_results = matching_hungarian(ocr_results, args.m3_labels)
        elif args.matching_algorithm == "greedy":
            new_ocr_results, ocr_results = matching_greedy(ocr_results, args.m3_labels, 100)
        else:
            raise NotImplementedError(f"matching algorithm {args.matching_algorithm}")
    return is_mission3, start_time, ocr_results, new_ocr_results, pers_mat

if __name__ == "__main__":
    from ref.EasyOCR.easyocr import Reader
    from configs.config_best import args
    from ocr_utils.ocr import put_all_text
    from ocr_utils.util import pickle_save
    reader = Reader(lang_list=args.langs, gpu=True, model_storage_directory="/home/jeonghokim/AGC_final/OCR/save_models", user_network_directory="/home/jeonghokim/AGC_final/OCR/save_models", detect_network=args.detector_name, recog_network=args.recognition_name, download_enabled=False)

    sample_img1 = cv2.imread("/home/data/AGC_final/sample_image/sample_cam1_01/000000.png")
    sample_img2 = cv2.imread("/home/data/AGC_final/sample_image/sample_cam2_01/000000.png")
    for n, sample_img in enumerate([sample_img1, sample_img2]):
        is_mission3, start_time, ocr_results, c_ocr_results, pers_mat = run_OCR(args=args, reader=reader, img=sample_img)
        ocr_img = put_all_text(sample_img, ocr_results)  # ocr 결과 시각화
        c_ocr_img = put_all_text(sample_img, c_ocr_results)  # ocr 결과 시각화
        os.makedirs(args.save_dir, exist_ok=True)
        cv2.imwrite(opj(args.save_dir, f"{n}_sample_img.png"), sample_img)
        cv2.imwrite(opj(args.save_dir, f"{n}_ocr_results.png"), ocr_img)
        cv2.imwrite(opj(args.save_dir, f"{n}_c_ocr_results.png"), c_ocr_img)
        pickle_save(opj(args.save_dir, f"{n}_ocr_results.pkl"), ocr_results)
        pickle_save(opj(args.save_dir, f"{n}_c_ocr_results.pkl"), c_ocr_results)
        if pers_mat is not None:
            np.save(opj(args.save_dir, f"{n}_pers_mat.npy"), pers_mat)
