import numpy as np

def NMS(c_pred_box, c_score,nms_thresh = 0.5):
    x1 = c_pred_box[:, 0]  #xmin
    y1 = c_pred_box[:, 1]  #ymin
    x2 = c_pred_box[:, 2]  #xmax
    y2 = c_pred_box[:, 3]  #ymax
    areas = (x2 - x1) * (y2 - y1)     
    order = c_score.argsort()[::-1]                

    keep = []                      
    while order.size > 0:
        i = order[0]                           
        keep.append(i)                    
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return keep

def postprocess(pred_box,score,conf_thresh = 0.5):
    cls_inds = np.argmax(score, axis=1)
    score = score[(np.arange(score.shape[0]), cls_inds)]
    keep = np.where(score >= conf_thresh)
    pred_box = pred_box[keep]
    score = score[keep]
    cls_inds = cls_inds[keep]
    # NMS
    keep = np.zeros(len(pred_box), dtype=np.int)
    for i in range(6):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_pred_box = pred_box[inds]
        c_score = score[inds]
        c_keep = NMS(c_pred_box, c_score)
        keep[inds[c_keep]] = 1
    keep = np.where(keep > 0)
    pred_box = pred_box[keep]
    score = score[keep]
    cls_inds = cls_inds[keep]
    return pred_box, score,cls_inds

