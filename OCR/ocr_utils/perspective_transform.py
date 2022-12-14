import cv2

import numpy as np

def get_pts(a,b,c,d):
    pts = np.array([a,b,c,d], dtype=np.float32)
    return pts
def perspective_transform(src_img, tr, br, rr, lr):
    """
    Args:
        src_img (np) : img [H x W x C]
        tr, br, rr, lr : top, bottom, right, left ratio
    Returns:
        perspective transformed image [H x W x C]
    """
    H, W, C = src_img.shape
    c_H = H//2
    c_W = W//2
    src_pts = get_pts([0,0], [0,H], [W,H], [W,0])
    dst_pts = get_pts([c_W*(1-tr),c_H*(1-rr)], [c_W*(1-br), c_H*(1+rr)], [c_W*(1+br), c_H*(1+lr)], [c_W*(1+tr), c_H*(1-lr)])
    pers = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dst_img = cv2.warpPerspective(src_img, pers, (W, H), flags=cv2.INTER_CUBIC)
    return dst_img, pers
def reverse_perspective_transform_coord(inv_pers_mat, coord):
    invw_x, invw_y, inv_w = inv_pers_mat@coord
    reverse_x = invw_x/inv_w
    reverse_y = invw_y/inv_w
    return [reverse_x, reverse_y]
        
    
    
