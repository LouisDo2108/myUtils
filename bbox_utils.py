import numpy as np
import copy

class Bbox:
    def __init__(self, img_h, img_w, bbox):
        self.img_h = img_h
        self.img_w = img_w
        self.bbox = bbox
        if len(bbox.shape) < 2:
            self.bbox = np.expand_dims(bbox, axis=0)
        
    def ccwh2xyxy(self):
        norm_center_x = self.bbox[:, 0]
        norm_center_y = self.bbox[:, 1]
        norm_label_width = self.bbox[:, 2]
        norm_label_height = self.bbox[:, 3]
        
        center_x = norm_center_x * self.img_w
        center_y = norm_center_y * self.img_h
        label_width = norm_label_width * self.img_w
        label_height = norm_label_height * self.img_h
        
        x_min = center_x - (label_width/2)
        y_min = center_y - (label_height/2)
        x_max = center_x + (label_width/2)
        y_max = center_y + (label_height/2)
    
    return np.array([x_min, y_min, x_max, y_max]).T

    def xyxy2xywh(self):
        bbox = copy.deepcopy(self.bbox)
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        return bbox
        