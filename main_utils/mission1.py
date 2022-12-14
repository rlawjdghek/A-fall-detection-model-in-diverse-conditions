import math
def l2_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_place(bbox, ocr_results):
    cx,cy,w,h = bbox
    best_d = math.inf
    for ocr_bbox, text, prob in ocr_results:
        tl, tr, br, bl = ocr_bbox 
        
        ocr_cx = int((tl[0] + br[0]) / 2)
        ocr_cy = int((tl[1] + br[1]) / 2)
        
        d = l2_distance(cx, cy, ocr_cx, ocr_cy)
        if d < best_d:
            best_d = d
            place = text
            best_ocr_cx = ocr_cx
            best_ocr_cy = ocr_cy
    return place, (best_ocr_cy, best_ocr_cy)
        

    