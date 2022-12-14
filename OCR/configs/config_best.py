import datetime
from pytz import timezone

class args:
    data_root_dir = "/home/data/AGC_final"  # 실제 코드에서 데이터 받을떄, 현 코드에서는 안씀
    langs = ["ko"]  # OCR 탐지 언어
    algorithm = "hungarian"  # ["hungarian", "greedy"]
    detector_name = "taejune"  # ["craft", "dbnet18", "taejune"]
    recognition_name = "20221013_taemo"  # ["korean_g2", "20221013_taemo"]
    m1_labels = ["글라스박스안경", "신한은행", "행복온누리약국", "만수무병", "김숙희반찬포유"]  # 미션 1 실제 레이블
    m3_labels = ['유리', '종이', '종이팩', '캔류', '페트', '플라스틱', '비닐']  # 미션 3 실제 레이블 
    #pt_ratio = [[0.5,1,1,1], [0.6,1,1,1], [0.7,1,1,1], [0.8,1,1,1], [0.5,1,0.8,1]]
    pt_ratio = [0.5,1,1,1]  # 추후 perspective 앙상블 할때 리스트로 
    pt_resize_r = 2 # 리사이징 크기
    m1_param_readtext = {"text_threshold": 0.75, "canvas_size": 6000, "width_ths":0.5, "mag_ratio": 1.75, "low_text": 0.4, "link_threshold": 0.2}  # ocr hyperparameter
    m3_param_readtext = {"text_threshold": 0.75, "canvas_size": 6000, "width_ths":0.5, "mag_ratio": 1.75, "low_text": 0.4, "link_threshold": 0.2}  # ocr hyperparameter 
    save_note = ""
    for (k1, v1), (k2, v2) in zip(m1_param_readtext.items(), m3_param_readtext.items()):
        save_note += f"_[{k1}-{v1},{v2}]"
    save_dir = f"/home/jeonghokim/AGC_final/OCR/save_images/{datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S') + save_note}"