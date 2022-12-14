import cv2

import torch
import torch.nn.functional as F
@torch.no_grad()
def m1_person_classification(model, imgs, transform, ensemble="soft"):
    """
    Args
        model : person classification model
        imgs : cropped numpy images, list
        transform : albumentation
    """
    t_imgs = []
    for img in imgs:
        t_img = transform(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]
        t_imgs.append(t_img)
    batch = torch.stack(t_imgs, dim=0).cuda()
    output = model.inference(batch)  # [BS x 3]
    if False:
        output_prob = F.softmax(output, dim=1)
        output_sum = output_prob.sum(dim=0)
        pred_val = output_sum.argmax(dim=0).item()
    else:
        pred_lst = output.argmax(dim=1).tolist()
        pred_val = max(pred_lst, key=pred_lst.count)
    return pred_val

    
    

    
    
    