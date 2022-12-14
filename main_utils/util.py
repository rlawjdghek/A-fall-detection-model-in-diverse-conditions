import argparse
from glob import glob
from os.path import join as opj

import numpy as np

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
# def get_current_time(start_time, frame_idx, sample_step):
#     h,m,s = start_time
#     total_s = (frame_idx * sample_step) // 15
#     delta_h = total_s // 3600
#     delta_m = (total_s % 3600) // 60
#     delta_s = total_s % 60
#     assert total_s == (delta_h * 3600 + delta_m * 60 + delta_s)
#     new_h = h + delta_h
#     new_m = m + delta_m
#     new_s = s + delta_s
#     if new_s >= 60:
#         new_s -= 60
#         new_m += 1
#     if new_m >= 60:
#         new_m -= 60
#         new_h += 1
#     if new_h >= 24:
#         new_h %= 24
#     return [new_h, new_m, new_s]
def get_current_time(start, frame_idx, step):
    h, m, s = map(int, start.split(":"))
    elapsed_frame = step * frame_idx
    elapsed_sec = elapsed_frame // 15

    s += elapsed_sec
    if s >= 60: m+=s//60; s%=60
    if m >= 60: h+=m//60; m%=60
    if h >= 24: h%=24
    return ":".join(list(map(str, [h, m, s])))
def matching_number(arr, ref_numbers):
    best_sum = -1
    for idx, ref_n in enumerate(ref_numbers):
        s = (arr == ref_n).sum()
        if best_sum < s:
            best_sum = s
            best_number = idx
    return str(best_number)
def extract_roi(img,th=230, h=973, dh=47, w=1644, dw=236, ddw=33, colon_w=19):
    img = img[:,:,::-1].copy()
    crop = img[h:h+dh, w:w+dw]    
    crop[crop<th] = 0
    crop[crop>=th] = 255
    crop = crop.any(axis=-1).astype(np.uint8)

    a1 = crop[:, :ddw]
    a2 = crop[:, ddw:ddw+ddw]
    c1 = crop[:, ddw+ddw:ddw+ddw+colon_w]
    a3 = crop[:, ddw+ddw+colon_w:ddw+ddw+colon_w+ddw]
    a4 = crop[:, ddw+ddw+colon_w+ddw:ddw+ddw+colon_w+ddw+ddw]
    c2 = crop[:, ddw+ddw+colon_w+ddw+ddw:ddw+ddw+colon_w+ddw+ddw+colon_w]
    a5 = crop[:, ddw+ddw+colon_w+ddw+ddw+colon_w:ddw+ddw+colon_w+ddw+ddw+colon_w+ddw]
    a6 = crop[:, ddw+ddw+colon_w+ddw+ddw+colon_w+ddw:ddw+ddw+colon_w+ddw+ddw+colon_w+ddw+ddw]

    return a1, a2, a3, a4, a5, a6    
def extract_time(img, ref_num_dir):
    ref_paths = sorted(glob(opj(ref_num_dir, "*.npy")))
    ref_numbers = [np.load(fp) for fp in ref_paths]
    h1_arr,h2_arr,m1_arr,m2_arr,s1_arr,s2_arr = extract_roi(img)
    h1 = matching_number(h1_arr, ref_numbers)
    h2 = matching_number(h2_arr, ref_numbers)
    m1 = matching_number(m1_arr, ref_numbers)
    m2 = matching_number(m2_arr, ref_numbers)
    s1 = matching_number(s1_arr, ref_numbers)
    s2 = matching_number(s2_arr, ref_numbers)
    return f"{h1+h2}:{m1+m2}:{s1+s2}"  

    