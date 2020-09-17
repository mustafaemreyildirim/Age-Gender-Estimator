from PIL import Image
from detector import detect_faces
from detectortoclass import Detector
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
from test import Predictor 
import os
from tqdm import tqdm
from visualization_utils import show_results
import argparse
import pdb

from docopt import docopt

def Align():
    d = Detector()
    p = Predictor()
    
    def align(img):

        agea = []
        gena = []
        crop_size = 64 # specify size of aligned faces, align and crop with padding
        scale = crop_size / 112.
        reference = get_reference_facial_points(default_square = True) * scale

        
        try:
            bounding_boxes, landmarks = d(img)
        except:
            bounding_boxes, landmarks = 0, 0
        total_im  = landmarks.shape[0]
    
        for i in range(total_im):
            facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j in range(5)]
                
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
                
            age , gender = p(img_warped)
            agea.append(int(age))
            if gender == 0.0:gender='Woman'
            else:gender = 'Man'
            gena.append(gender)

        return show_results(img,agea,gena,bounding_boxes,landmarks)
    return align