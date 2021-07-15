"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import fnmatch
import numpy as np
import cv2
import csv
import os
from collections import OrderedDict

from mrcnn import config
from mrcnn import utils



class ModelConfig(config.Config):
    
    NAME = "Cardiomegaly"  # Override in sub-classes

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200

    BACKBONE = "resnet101"

    RPN_TRAIN_ANCHORS_PER_IMAGE = 32
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    
    IMAGE_CHANNEL_COUNT = 3
    
    DETECTION_MIN_CONFIDENCE = 0.05
    DETECTION_NMS_THRESHOLD = 0.0001 # 0: no overlap in one class.

    
    def __init__(self):
        """Set values of computed attributes."""
        self.Classes = {'Aortic Knob': 1,
           'Carina': 2,
           'DAO': 3,
           'LAA': 4,
           'Lt Lower CB': 5,
           'Pulmonary Conus': 6,
           'Rt Lower CB': 7,
           'Rt Upper CB': 8}
        
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        for classname in self.Classes:
            self.add_class("Cardiomegaly", self.Classes[classname], classname)
        
        self.NUM_CLASSES = 1+len(self.Classes)
        self.STEPS_PER_EPOCH = self.STEPS_PER_EPOCH / self.IMAGES_PER_GPU
        self.VALIDATION_STEPS = self.VALIDATION_STEPS / self.IMAGES_PER_GPU
        super(ModelConfig, self).__init__()

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })    
        
        
