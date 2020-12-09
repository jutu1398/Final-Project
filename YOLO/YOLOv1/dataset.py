import os

import torch
from torchvision import datasets

from boxlist import BoxList


def has_only_empty_bbox(annot):
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


def has_valid_annotation(annot):
    if len(annot) == 0:
        return False

    if has_only_empty_bbox(annot):
        return False

    return True


class COCODataset(datasets.CocoDetection):
    def __init__(self, path, split, transform=None, S = 7, B = 2, C = 20):
        root = os.path.join(path, f'{split}2017')
        annot = os.path.join(path, 'annotations', f'instances_{split}2017.json')
        self.S = S
        self.B = B
        self.C = C
        
        
        super().__init__(root, annot)

        self.ids = sorted(self.ids)

        if split == 'train':
            ids = []

            for id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=id, iscrowd=None)
                annot = self.coco.loadAnns(ann_ids)
               
                if has_valid_annotation(annot):
                    ids.append(id)

            self.ids = ids

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.id2img = {k: v for k, v in enumerate(self.ids)}

        self.transform = transform
        
    def encode(self, boxes, labels, img):
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image    
            width/height.
            labels: (tensor) [c_obj1, c_obj2, ...]
        Returns:
            An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        """
        boxes = boxes.box
        #print(img)
        _, h, w = img
        boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes) # normalize (x1, y1, x2, y2) w.r.t. image         width/height.
        #print(boxes)
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C

        target = torch.zeros(S, S, N)
        cell_size = 1.0 / float(S)
        boxes_wh = boxes[:, 2:] - boxes[:, :2] # width and height for each box, [n, 2]
        boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0 # center x & y for each box, [n, 2]
        #print(boxes_wh, boxes_xy)
        for b in range(boxes.size(0)):
            xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])
            
            ij = (xy / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1]) # y & x index which represents its location on the grid.
            x0y0 = ij * cell_size # x & y of the cell left-top corner.
            xy_normalized = (xy - x0y0) / cell_size # x & y of the box on the cell, normalized from 0.0 to 1.0
            #print(xy, ij, i, j, boxes.size(0))
            # TBM, remove redundant dimensions from target tensor.
            # To remove these, loss implementation also has to be modified.
            for k in range(B):
                #print(label)
                s = 5 * k
                target[j, i, s  :s+2] = xy_normalized
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0
            #print(target.shape)
            target[j, i, 5*B + label - 1] = 1.0

        return target

    def __getitem__(self, index):
        img, annot = super().__getitem__(index)

        annot = [o for o in annot if o['iscrowd'] == 0]

        boxes = [o['bbox'] for o in annot]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        target = BoxList(boxes, img.size, mode='xywh').convert('xyxy')
        
        
        classes = [o['category_id'] for o in annot]
        classes = [self.category2id[c] for c in classes]
        classes = torch.tensor(classes)
        target.fields['labels'] = classes
        
        target.clip(remove_empty=True)
        
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        target = self.encode(target, classes, img.shape)
        return img, target, classes, index

    def get_image_meta(self, index):
        id = self.id2img[index]
        img_data = self.coco.imgs[id]

        return img_data


class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = (max_size[1] | (stride - 1)) + 1
        max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)
