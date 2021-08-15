
import random
import glob
import paddle
import numpy as np
import xml.etree.ElementTree as ET


def iou(bbox, priors):
    """
    计算一个真实框与 k个先验框(priors) 的交并比(IOU)值。
    bbox: 真实框的宽高，数据为（宽，高），其中宽与高都是归一化后的相对值。
    priors: 生成的先验框，数据形状为(k,2)，其中k是先验框priors的个数
    """
    x = np.minimum(priors[:, 0], bbox[0])
    y = np.minimum(priors[:, 1], bbox[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("真实框有误")
    intersection = x * y
    bbox_area = bbox[0] * bbox[1]
    priors_area = priors[:, 0] * priors[:, 1]
    iou_ = intersection / (bbox_area + priors_area - intersection)
    return iou_


def avg_iou(bboxes, priors):
    """
    计算一个真实框和k个先验框的IOU的均值。
    """
    return np.mean([np.max(iou(bboxes[i], priors)) for i in range(bboxes.shape[0])])

def kmeans(bboxes, k, dist=np.median):
    """
    利用IOU值进行K-means聚类
    bboxes: 真实框，数据形状为(n, 2)，其中n是真实框的个数，2表示宽与高
    k: 聚类中心个数，此处表示要生成的先验框的个数
    dist: 用于更新聚类中心的函数，默认为求中值
    返回 k 个先验框priors ， 数据形状为（k，2）
    """
    # 获取真实框个数
    n = bboxes.shape[0]
    # 距离数组，记录每一个真实框和 k 个先验框的距离
    distances = np.empty((n, k))
    # 记录每一个真实框属于哪一个聚类中心，即与哪一个先验框的IOU值最大，记录的是先验框的索引值
    last_priors = np.zeros((n,))

    # 初始化聚类中心，随机从n个真实框中选择k个框作为聚类中心priors
    np.random.seed()
    priors = bboxes[np.random.choice(n, k, replace=False)]

    while True:
        # 计算每一个真实框和k个先验框的距离，距离指标为 1-IOU(box,priors)
        for i in range(n):
            distances[i] = 1 - iou(bboxes[i], priors)
        # 对于每一个真实框，要选取对应着 distances 最小的那个先验框，获取索引值
        nearest_priors = np.argmin(distances, axis=1)
        # 如果获取到的索引值没变，说明聚类结束
        if (last_priors == nearest_priors).all():
            break
        # 更新聚类中心
        for j in range(k):
            priors[j] = dist(bboxes[nearest_priors == j], axis=0)
        # 更新last_priors
        last_priors = nearest_priors
    return priors

def load_dataset(path):
    '''
    path: 标注文件，xml文件所在文件夹路径
    '''
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)
        # 图片高度
        height = int(tree.findtext("./size/height"))
        # 图片宽度
        width = int(tree.findtext("./size/width"))
        
        for obj in tree.iter("object"):
            # 相对值
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height
            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            if xmax == xmin or ymax == ymin:
                print(xml_file)
            dataset.append([xmax - xmin, ymax - ymin]) # 宽与高的相对值
    return np.array(dataset) # 转为numpy数组



def bbox2tensor(bbox,max_num=30):
    '''
    bbox：标签信息。信息格式为[cls,x,y,w,h, cls,x,y,w,h, cls,x,y,w,h] 每5个元素为一组标签信息
    max_num: 一张图片中最大的目标数，默认最多只能有30个物体
    返回标签信息，tensor
    '''

    gt_bbox = paddle.zeros(shape=[max_num, 5], dtype='float32')
    for i in range(len(bbox)//5):
        gt_bbox[i, 0] = bbox[i*5]
        gt_bbox[i, 1] = bbox[i*5+1]
        gt_bbox[i, 2] = bbox[i*5+2]
        gt_bbox[i, 3] = bbox[i*5+3]
        gt_bbox[i, 4] = bbox[i*5+4]
        if i >= max_num:
            break
    return gt_bbox

def calculate_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0


# 将网络输出的[tx, ty, th, tw]转化成预测框的坐标[x1, y1, x2, y2]
def get_yolo_box_xxyy(pred, anchors, num_classes):
    """
    将网络输出的[tx, ty, th, tw]转化成预测框的坐标[x1, y1, x2, y2]，也就是 [左上角坐标，右上角坐标] 格式
    pred：网络输出，tensor
    anchors： 是一个list。表示锚框的大小。
            YOLOv2官方配置文件中，anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]，
            表示有5个锚框，第一个锚框大小[w, h]是[0.57273, 0.677385]，第5个锚框大小是[9.77052, 9.16828]
            锚框的大小都是表示在特征图13x13中的大小
    num_classes：类别数

    返回预测框pred_box, 最终输出数据保存在pred_box中，其形状是[N, num_anchors, 4, H, W,]，4表示4个位置坐标
    """
    num_rows = pred.shape[-2]
    num_cols = pred.shape[-1]
    num_anchors = len(anchors) // 2

    # pred的形状是[batchsize, C, H, W]，其中C = num_anchors * (5 + num_classes)
    # 对pred进行reshape
    pred = pred.reshape([-1, num_anchors, 5 + num_classes, num_rows, num_cols])
    pred = paddle.transpose(pred, perm=[0, 3, 4, 1, 2])
    # 取出与位置相关的数据
    pred_location = pred[:, :, :, :, 0:4]

    anchors_this = []
    for ind in range(num_anchors):
        anchors_this.append([anchors[ind * 2], anchors[ind * 2 + 1]])
    # anchors_this = np.array(anchors_this).astype('float32')
    anchors_this = paddle.to_tensor(anchors_this)

    pred_box = paddle.zeros(pred_location.shape)
    # for b in range(batchsize):
    for i in range(num_rows):
        for j in range(num_cols):
            for k in range(num_anchors):
                pred_box[:, i, j, k, 0] = j  # 列
                pred_box[:, i, j, k, 1] = i  # 行
                pred_box[:, i, j, k, 2] = anchors_this[k][0]  # 先验框宽
                pred_box[:, i, j, k, 3] = anchors_this[k][1]  # 先验框高

    # 这里使用相对坐标，pred_box的输出元素数值在0.~1.0之间, 相对于特征图大小的相对值
    pred_box[:, :, :, :, 0] = (paddle.nn.functional.sigmoid(pred_location[:, :, :, :, 0]) + pred_box[:, :, :, :, 0]) / num_cols
    pred_box[:, :, :, :, 1] = (paddle.nn.functional.sigmoid(pred_location[:, :, :, :, 1]) + pred_box[:, :, :, :, 1]) / num_rows
    pred_box[:, :, :, :, 2] = paddle.exp(pred_location[:, :, :, :, 2]) * pred_box[:, :, :, :, 2] / num_cols
    pred_box[:, :, :, :, 3] = paddle.exp(pred_location[:, :, :, :, 3]) * pred_box[:, :, :, :, 3] / num_rows

    # 将坐标从xywh转化成xyxy，也就是 [左上角坐标，右上角坐标] 格式
    pred_box[:, :, :, :, 0] = pred_box[:, :, :, :, 0] - pred_box[:, :, :, :, 2] / 2. 
    pred_box[:, :, :, :, 1] = pred_box[:, :, :, :, 1] - pred_box[:, :, :, :, 3] / 2. 
    pred_box[:, :, :, :, 2] = pred_box[:, :, :, :, 0] + pred_box[:, :, :, :, 2]
    pred_box[:, :, :, :, 3] = pred_box[:, :, :, :, 1] + pred_box[:, :, :, :, 3]

    return pred_box

# 获取标签
def get_label(pred, gt_bboxs, anchors, iou_threshold, step_less_12800, num_classes=6, rescore=False):
    '''
    pred：网络输出
    gt_bboxs: 真实框信息，[class,x,y,w,h]，其中x,y,w,h为归一化后的数据
    anchors： 是一个list。表示锚框的大小。
            YOLOv2官方配置文件中，anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]，（coco）
            表示有5个锚框，第一个锚框大小[w, h]是[0.57273, 0.677385]，第5个锚框大小是[9.77052, 9.16828]
            锚框的大小都是表示在特征图13x13中的大小
    step_less_12800：训练步数是否小于12800，bool
    num_classes：类别数
    返回：
    label_objectness_confidence,label_location,label_classification,scale_location,object_mask,noobject_mask
    '''

    batchsize, _, h, w = pred.shape  # h = w = 13 特征图大小13x13
    _, nums, c = gt_bboxs.shape  # nums 表示一张图中最多只能由nums个目标，c = 5
    num_anchors = len(anchors) // 2  # num_anchors = 5

    pred_box = get_yolo_box_xxyy(pred, anchors, num_classes)  # 获取预测框，此时预测框中的坐标格式为（左上角坐标，右上角坐标）
    # 形状为[batchsize, 13, 13, num_anchors, 4]
    pred_box = pred_box.numpy()
    gt_bboxs = gt_bboxs.numpy()  # shape = (batchsize, nums, 5)
    anchors_copy = np.array(anchors).reshape((num_anchors, 2))  # shape = (num_anchors,2)
    anchors_copy = np.expand_dims(anchors_copy, 0).repeat(batchsize, axis=0)  # shape = (batchsize, num_anchors,2)
    # print(anchors_copy.shape)
    # print(anchors_copy)

    label_objectness_confidence = np.zeros(shape=(batchsize, h, w, num_anchors), dtype='float32')
    label_location = np.zeros(shape=(batchsize, h, w, num_anchors, 4), dtype='float32')
    label_classification = np.zeros(shape=(batchsize, h, w, num_anchors, num_classes), dtype='float32')
    scale_location = 0.01 * np.ones((batchsize, h, w, num_anchors), dtype='float32')  # 与位置损失相关的权重系数
    object_mask = np.zeros(shape=(batchsize, h, w, num_anchors), dtype='float32')  # 有目标掩码
    noobject_mask = np.ones(shape=(batchsize, h, w, num_anchors), dtype='float32')  # 无目标掩码

    # 对于不负责预测目标的预测框，如果其与真实框的IOU大于iou_threshold（默认0.6），此预测框不参与任何损失计算
    iou_above_thresh_indices = np.zeros((batchsize, h, w, num_anchors))

    # 训练步数小于12800时，需要计算预测框与先验框的位置损失
    if (step_less_12800):
        label_location[:, :, :, :, 0] = 0.5
        label_location[:, :, :, :, 1] = 0.5

    gt_cls = gt_bboxs[:, :, 0].astype(np.int32)  # shape = (batchsize , nums,)
    gt_center_x = gt_bboxs[:, :, 1]  # shape = (batchsize * nums)
    gt_center_y = gt_bboxs[:, :, 2]
    gt_w = gt_bboxs[:, :, 3]  # shape = (batchsize , nums,)
    gt_h = gt_bboxs[:, :, 4]
    gtx_min = gt_center_x - gt_w / 2.0
    gtx_max = gt_center_x + gt_w / 2.0
    gty_min = gt_center_y - gt_h / 2.0
    gty_max = gt_center_y + gt_h / 2.0

    target_indexs = np.where(gt_bboxs[:, :, 3] != 0)  # 宽不为0的目标框(真实目标)的索引值, [batch][num]
    i_float = gt_center_y * h
    i = np.floor(i_float).astype(np.int32)  # 在第几行  shape = (batchsize, nums,)
    i = i[target_indexs]  # 取出对应i
    j_float = gt_center_x * w  # shape = (batchsize, nums,)
    j = np.floor(j_float).astype(np.int32)  # 在第几列
    j = j[target_indexs]  # 取出对应j

    gt_bboxs_copy = np.expand_dims(gt_bboxs, 1).repeat(h * w * num_anchors, axis=1)  # shape = (batchsize, h*w*num_anchors, nums, 5)
    gt_bboxs_copy = gt_bboxs_copy.reshape((batchsize, h, w, num_anchors, nums, 5))[:, :, :, :, :, 1:]  # shape = (batchsize, h, w,num_anchors, nums, 5)
    gtx_min_copy = gt_bboxs_copy[:, :, :, :, :, 0] - gt_bboxs_copy[:, :, :, :, :, 2] / 2.  # shape = (batchsize, h, w,num_anchors, nums)
    gty_min_copy = gt_bboxs_copy[:, :, :, :, :, 1] - gt_bboxs_copy[:, :, :, :, :, 3] / 2.
    gtx_max_copy = gt_bboxs_copy[:, :, :, :, :, 0] + gt_bboxs_copy[:, :, :, :, :, 2] / 2.
    gty_max_copy = gt_bboxs_copy[:, :, :, :, :, 1] + gt_bboxs_copy[:, :, :, :, :, 3] / 2.

    ious = []
    for a in range(num_anchors):
        bbox1 = np.zeros((batchsize, nums, 4))  # 将真实框的中心点移到原点
        bbox1[:, :, 2] = gt_w  # gt_w.shape = (batchsize,nums,)
        bbox1[:, :, 3] = gt_h  # shape = (batchsize,nums,)
        anchor_w = anchors[a * 2]
        anchor_h = anchors[a * 2 + 1]

        # x1 = np.maximum(bbox1[:, :, 0], 0) # x1.shape = (batchsize,nums,)
        # y1 = np.maximum(bbox1[:, :, 1], 0)
        x2 = np.minimum(bbox1[:, :, 2], anchor_w)  # x2.shape = (batchsize,nums,)
        y2 = np.minimum(bbox1[:, :, 3], anchor_h)
        intersection = np.maximum(x2, 0.) * np.maximum(y2, 0.)  # intersection.shape = (batchsize,nums,)
        s1 = gt_w * gt_h
        s2 = anchor_w * anchor_h
        union = s2 + s1 - intersection
        iou = intersection / union  # iou.shape = (batchsize,nums,)
        ious.append(iou)
    ious = np.array(ious)  # ious.shape = (num_anchors,batchsize,nums)
    inds_anchor = np.argmax(ious, axis=0)  # inds.shape = (batchsize,nums,)  # 获取与目标真实框IOU最大的anchor索引值
    inds_anchor = inds_anchor[target_indexs].astype(np.int32)  # 取出对应anchor索引值

    # 设置掩码
    object_mask[target_indexs[0], i, j, inds_anchor] = 1.  # 把掩码中的对应位置设为1
    noobject_mask[target_indexs[0], i, j, inds_anchor] = 0  # 把掩码中的对应位置设为

    # 设置位置标签
    # 对于负责预测目标的预测框, 需要计算位置损失
    dx_label = j_float[target_indexs] - j  # x方向上的偏移量，tx的标签值
    dy_label = i_float[target_indexs] - i  # y方向上的偏移量，ty的标签值
    dw_label = np.log(j_float[target_indexs] / anchors_copy[target_indexs[0], inds_anchor, 0])  # tw的标签值
    dh_label = np.log(i_float[target_indexs] / anchors_copy[target_indexs[0], inds_anchor, 1])  # th的标签值
    label_location[target_indexs[0], i, j, inds_anchor, 0] = dx_label
    label_location[target_indexs[0], i, j, inds_anchor, 1] = dy_label
    label_location[target_indexs[0], i, j, inds_anchor, 2] = dw_label
    label_location[target_indexs[0], i, j, inds_anchor, 3] = dh_label

    # scale_location用来调节不同尺寸的锚框对损失函数的贡献，作为加权系数与位置损失函数相乘
    scale_location[target_indexs[0], i, j, inds_anchor] = 2.0 - gt_w[target_indexs] * gt_h[target_indexs]
    # 设置类别标签
    c = gt_cls[target_indexs]
    label_classification[target_indexs[0], i, j, inds_anchor, c] = 1.

    # 设置置信度标签
    if rescore:
        # 计算对应的预测框与真实框之间的IOU值
        bbox_pred_xyxy = pred_box[target_indexs[0], i, j, inds_anchor, :]
        # bbox_gt_xyxy = np.zeros(bbox_pred_xyxy.shape)
        # bbox_gt_xyxy[:,0] =  gtx_min[target_indexs]
        # bbox_gt_xyxy[:,1] =  gty_min[target_indexs]
        # bbox_gt_xyxy[:,2] =  gtx_max[target_indexs]
        # bbox_gt_xyxy[:,3] =  gty_max[target_indexs]
        x1 = np.maximum(bbox_pred_xyxy[:, 0], gtx_min[target_indexs])  # x1.shape = (batchsize,nums,)
        y1 = np.maximum(bbox_pred_xyxy[:, 1], gty_min[target_indexs])
        x2 = np.minimum(bbox_pred_xyxy[:, 2], gtx_max[target_indexs])
        y2 = np.minimum(bbox_pred_xyxy[:, 3], gty_max[target_indexs])
        intersection = np.maximum(x2 - x1, 0.) * np.maximum(y2 - y1, 0.)
        s1 = gt_w[target_indexs] * gt_h[target_indexs]
        s2 = (bbox_pred_xyxy[:, 2] - bbox_pred_xyxy[:, 0]) * (bbox_pred_xyxy[:, 3] - bbox_pred_xyxy[:, 1])
        union = s2 + s1 - intersection
        iou = intersection / union
        # 对于负责预测目标的预测框，置信度标签为预测框与真实框之间IOU值，（也可直接设置为1，官方源码中默认用iou）
        label_objectness_confidence[target_indexs[0], i, j, inds_anchor] = iou
    else:
        label_objectness_confidence[target_indexs[0], i, j, inds_anchor] = 1

    # 对于不负责预测目标的预测框，如果其与真实框的IOU大于iou_threshold（默认0.6），此预测框不参与任何损失计算
    for num in range(nums):
        x1 = np.maximum(pred_box[:, :, :, :, 0], gtx_min_copy[:, :, :, :, num])  # x1.shape = (batchsize, h, w,num_anchors,)
        y1 = np.maximum(pred_box[:, :, :, :, 1], gty_min_copy[:, :, :, :, num])
        x2 = np.minimum(pred_box[:, :, :, :, 2], gtx_max_copy[:, :, :, :, num])
        y2 = np.minimum(pred_box[:, :, :, :, 3], gty_max_copy[:, :, :, :, num])
        intersection = np.maximum(x2 - x1, 0.) * np.maximum(y2 - y1, 0.)  # intersection.shape = (batchsize, h, w,num_anchors)
        s1 = (gtx_max_copy[:, :, :, :, num] - gtx_min_copy[:, :, :, :, num]) * (
                    gty_max_copy[:, :, :, :, num] - gty_min_copy[:, :, :, :, num])
        s2 = (pred_box[:, :, :, :, 2] - pred_box[:, :, :, :, 0]) * (pred_box[:, :, :, :, 3] - pred_box[:, :, :, :, 1])
        union = s2 + s1 - intersection + 1.0e-10  # union.shape = (batchsize, h, w,num_anchors)
        iou = intersection / union  # iou.shape = (batchsize, h, w,num_anchors)
        above_inds = np.where(iou > iou_threshold)
        iou_above_thresh_indices[above_inds] = 1  # 大于iou_threshold的对应位置设置为1
    negative_indices = (label_objectness_confidence == 0)
    ignore_indices = negative_indices * iou_above_thresh_indices.astype('bool')
    noobject_mask[ignore_indices] = 0  # 对于不负责预测目标的预测框，如果其与真实框的IOU大于iou_threshold（默认0.6），此预测框不参与任何损失计算

    # 转为tensor
    label_objectness_confidence = paddle.to_tensor(label_objectness_confidence)
    label_location = paddle.to_tensor(label_location)
    label_classification = paddle.to_tensor(label_classification)
    scale_location = paddle.to_tensor(scale_location)
    object_mask = paddle.to_tensor(object_mask)
    noobject_mask = paddle.to_tensor(noobject_mask)

    return label_objectness_confidence, label_location, label_classification, scale_location, object_mask, noobject_mask


