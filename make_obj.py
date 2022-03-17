import sys
import argparse
import cv2
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool

### To use CPU
import os
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX


config_path = 'configs/mb1_120x120.yml'
data_base_path = 'data'
opt = 'obj'

image_path = os.path.join(data_base_path, 'target_images')

cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'

face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)

png_filename_list = [
    os.path.join(image_path, filename) 
        for filename in os.listdir(image_path) 
            if filename.endswith('.png')
]

for png_filename in png_filename_list: 
    # Given a still image path and load to BGR channel
    img = cv2.imread(png_filename)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = True
    # old_suffix = get_suffix(png_filename)
    new_suffix = '.{}'.format(opt)

    result_path = os.path.join(data_base_path, 'obj_results')
    os.makedirs(result_path, exist_ok=True)

    # wfp = f'{result_path}/{image_path.split("/")[-1].replace(old_suffix, "")}_{opt}' + new_suffix
    basename = os.path.basename(png_filename)
    name = os.path.splitext(basename)[0]
    wfp = os.path.join(result_path, name+new_suffix)

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)

