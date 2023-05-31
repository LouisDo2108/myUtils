import numpy as np
import copy


class Bbox:
    '''
    This class is used to convert bounding box format.
    The original bounding box format is [x1, y1, x2, y2].
    '''
    def __init__(self, img_h, img_w, bbox, format='xyxy'):
        self.img_h = img_h
        self.img_w = img_w
        self.bbox = np.asarray(bbox)
        self.format = format
        
        if len(self.bbox.shape) < 2:
            self.bbox = np.expand_dims(self.bbox, axis=0)
        
    def ccwh2xyxy(self):
        if self.format == 'ccwh':
            norm_center_x, norm_center_y, norm_label_width, norm_label_height = np.split(self.bbox, 4, axis=1)
            
            center_x = norm_center_x * self.img_w
            center_y = norm_center_y * self.img_h
            label_width = norm_label_width * self.img_w
            label_height = norm_label_height * self.img_h
            
            x_min = center_x - (label_width / 2)
            y_min = center_y - (label_height / 2)
            x_max = center_x + (label_width / 2)
            y_max = center_y + (label_height / 2)
        
            return np.concatenate([x_min, y_min, x_max, y_max], axis=1)
        else:
            raise ValueError("The current format is not 'ccwh'. Cannot perform conversion.")
    
    def xyxy2xywh(self):
        if self.format == 'xyxy':
            bbox = copy.deepcopy(self.bbox)
            bbox[:, 2] -= bbox[:, 0]  # Calculate width
            bbox[:, 3] -= bbox[:, 1]  # Calculate height
            return bbox
        else:
            raise ValueError("The current format is not 'xyxy'. Cannot perform conversion.")

    def xyxy2cxcywh(self):
        if self.format == 'xyxy':
            bbox = self.xyxy2xywh()
            bbox[:, 0] += bbox[:, 2] / 2  # Calculate center x
            bbox[:, 1] += bbox[:, 3] / 2  # Calculate center y
            return bbox
        else:
            raise ValueError("The current format is not 'xyxy'. Cannot perform conversion.")