#encoding:utf-8
#
#created by xiongzihua
#
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 7
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
            for j in range(grid_num):
                for b in range(2):
                    index = min_index[i,j]
                    mask[i,j,index] = 0
                    if mask[i,j,b] == 1:
                        #print(i,j,b)
                        box = pred[i,j,b*5:b*5+4]
                        contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                        #xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
    #                     box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                        box_xy = torch.FloatTensor(box.size())
                        box_xy[:2] = box[:2] - 0.5*box[2:]
                        box_xy[2:] = box[:2] + 0.5*box[2:]
                        max_prob,cls_index = torch.max(pred[i,j,10:],0)
                        if float((contain_prob*max_prob)[0]) > 0.1:
                            boxes.append(box.view(1,4))
                            #boxes.append(box_xy.view(1,4))
                            cls_indexs.append(cls_index)
                            probs.append(contain_prob*max_prob)
                            #print(i,j,b, box.view(1,4) )
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.stack(cls_indexs) #(n,)
    keep = nms(boxes,probs)
    #print("nms:", len(keep), boxes[keep])
    return boxes[keep],cls_indexs[keep],probs[keep]
    #return boxes,cls_indexs,probs

def nms(boxes, scores):
        """ Apply non maximum supression.
        Args:
        Returns:
        """
        
        #print(boxes, boxes.size())
        threshold = .3

        x1 = boxes[:, 0] # [n,]
        y1 = boxes[:, 1] # [n,]
        x2 = boxes[:, 2] # [n,]
        y2 = boxes[:, 3] # [n,]
        areas = (x2 - x1) * (y2 - y1) # [n,]

        _, ids_sorted = scores.sort(0, descending=True) # [n,]
        ids = []
        while ids_sorted.numel() > 0:
            # Assume `ids_sorted` size is [m,] in the beginning of this iter.

            i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
            ids.append(i)

            if ids_sorted.numel() == 1:
                break # If only one box is left (i.e., no box to supress), break.

            inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i]) # [m-1, ]
            inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i]) # [m-1, ]
            inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i]) # [m-1, ]
            inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i]) # [m-1, ]
            inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

            inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
            unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
            ious = inters / unions # [m-1, ]

            # Remove boxes whose IoU is higher than the threshold.
            ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
            if ids_keep.numel() == 0:
                break # If no box left, break.
            ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

        return torch.LongTensor(ids)
# def nms(bboxes,scores,threshold=0.5):
#     '''
#     bboxes(tensor) [N,4]
#     scores(tensor) [N,]
#     '''
#     print(bboxes, bboxes.size())
#     x1 = bboxes[:,0]
#     y1 = bboxes[:,1]
#     x2 = bboxes[:,2]
#     y2 = bboxes[:,3]
    
#     areas = (x2-x1) * (y2-y1)

#     _,order = scores.sort(0,descending=True)
#     keep = []
#     order = order.tolist()
#     while len(order) > 0:
#         print(order)
#         #print(len(order))
#         i = order[0]
#         keep.append(i)

#         if len(order) == 1:
#             break

#         xx1 = x1[order[1:]].clamp(min=x1[i])
#         yy1 = y1[order[1:]].clamp(min=y1[i])
#         xx2 = x2[order[1:]].clamp(max=x2[i])
#         yy2 = y2[order[1:]].clamp(max=y2[i])

#         w = (xx2-xx1).clamp(min=0)
#         h = (yy2-yy1).clamp(min=0)
#         inter = w*h

#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#         print(ovr)
#         ids = (ovr<=threshold).nonzero().squeeze()
#         ids = ids.tolist()
#         if type(ids) != "int":
#             break
#         order = order[ids+1]
#     return torch.LongTensor(keep)
#
#start predict one image
#
def predict_gpu(model,image_name,root_path=''):

    result = []
    image = cv2.imread(root_path+image_name)
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    img = Variable(img[None,:,:,:],volatile=True)
    img = img.cuda()

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result
        



# if __name__ == '__main__':
#     model = resnet50()
#     print('load model...')
#     model.load_state_dict(torch.load('best.pth'))
#     model.eval()
#     model.cuda()
#     image_name = 'dog.jpg'
#     image = cv2.imread(image_name)
#     print('predicting...')
#     result = predict_gpu(model,image_name)
#     for left_up,right_bottom,class_name,_,prob in result:
#         color = Color[VOC_CLASSES.index(class_name)]
#         cv2.rectangle(image,left_up,right_bottom,color,2)
#         label = class_name+str(round(prob,2))
#         text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
#         p1 = (left_up[0], left_up[1]- text_size[1])
#         cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
#         cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

#     cv2.imwrite('result.jpg',image)
