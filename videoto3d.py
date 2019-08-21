from __future__ import absolute_import, division, print_function

import os, re, sys, subprocess, glob, time
from humanfriendly import format_timespan
from math import floor

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import io
from PIL import Image
from addict import Dict

from libs.models import *
from libs.utils import DenseCRF

############################
##    GLOBAL VARIABLES    ##
############################

CONFIG_PATH = ["E:\\videoto3d\\deeplab-pytorch\\configs\\cocostuff10k.yaml",
               "E:\\videoto3d\\deeplab-pytorch\\configs\\cocostuff164k.yaml",
               "E:\\videoto3d\\deeplab-pytorch\\configs\\voc12.yaml"]

MODEL_PATH = ["E:\\videoto3d\\trained_models\\cocostuff10k\\checkpoint_final.pth",
              "E:\\videoto3d\\trained_models\\cocostuff164k\\checkpoint_final.pth",
              "E:\\videoto3d\\trained_models\\voc12\\checkpoint_final.pth"]

MODEL_NAME = ["COCOStuff10k",
              "COCOStuff164k",
              "VOC12"]

DATA_PATH = "E:\\videoto3d\\media"
VIDEOS_PATH = DATA_PATH + "\\videos"
SETS_PATH = DATA_PATH + "\\sets"

PHOTOSCAN_PATH = 'C:\\Program Files\\Agisoft\\PhotoScan Pro'
PHOTOSCAN_EXE = PHOTOSCAN_PATH + '\\photoscan.exe'
LICENSE_EXE = PHOTOSCAN_PATH + '\\rlm.exe'

PHOTOSCAN_SCRIPT_PATH = 'E:\\videoto3d\\get3dmodel.py'

CRF = True
CUDA = True



############################
##     MORE FUNCTIONS     ##
############################

def create_directory(path):
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)

def directory_setup():
    create_directory(DATA_PATH)
    create_directory(VIDEOS_PATH)
    create_directory(SETS_PATH)

def check_video(video_path):
    videos_path = VIDEOS_PATH + '\\*.mp4'
    print(video_path)
    print('Checking if video is in directory...')

    for video in glob.glob(videos_path):
        print(video)
        if video.lower() == video_path.lower():
            print('- Video found')
            return True
    
    sys.exit('- Video not found in directory. Exiting program')
        

############################
##      SEGMENTATION      ##
############################

def select_model():
    model = -1
    print('- Which segmentation model would you like to use?')
    print('\t0. COCOStuff 10k')
    print('\t1. COCOStuff 164k')
    print('\t2. PASCAL VOC 12')
    
    while model < 0 or model > 2:
        model = int(input('\t>> Select segmentation model number: '))
    
    return model

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor

def preprocessing(image, device, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap

def select_object(labels):
    selection = -1
    selection = int(input('Select object: '))
    for i, label in enumerate(labels):
        if selection == label:
            return selection
    print('Selection not valid')

def process_mask(mask):
    kernel5 = np.ones((5,5),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)
    kernel10 = np.ones((10,10),np.uint8)
    mask = cv2.dilate(mask,kernel5,iterations = 10)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel10, iterations = 10) 
    mask = cv2.erode(mask,kernel3,iterations = 5)
    return mask

def generate_image_from_mask(mask,width,height):
    im_mask = Image.fromarray(mask * 255)
    im_mask = im_mask.resize((width,height),Image.NEAREST)
    im_mask_grey = im_mask.convert('RGB')
    im_mask_grey = cv2.cvtColor(np.float32(im_mask_grey), cv2.COLOR_RGB2GRAY)
    (thresh, im_mask_bw) = cv2.threshold(im_mask_grey, 254, 255, cv2.THRESH_BINARY)
    im_mask_bw = process_mask(im_mask_bw)
    return im_mask_bw

def get_label(image, model):
    print('Loading segmentation configuration...')
    config_path = CONFIG_PATH[model]
    print(config_path)
    model_path = MODEL_PATH[model]
    print(model_path)

    config_path = io.open(config_path, 'r')
    CONFIG = Dict(yaml.load(config_path))
    device = get_device(CUDA)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if CRF else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    height, width = image.shape[:2]
    print('Image size: ' + str(width) + 'x' + str(height))
    # inference
    image, raw_image = preprocessing(image, device, CONFIG)
    labelmap = inference(model, image, raw_image, postprocessor)
    labels = np.unique(labelmap)

    # Show result for each class
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(rows, cols, 1)
    ax.set_title("Input image")
    ax.imshow(raw_image[:, :, ::-1])
    ax.axis("off")

    print('- Object list: ')
    for i, label in enumerate(labels):
        print('    ',label,':',classes[label])
        mask = labelmap == label
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title(classes[label])
        ax.imshow(raw_image[..., ::-1])
        ax.imshow(mask.astype(np.float32), alpha=0.5)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    selected_object = select_object(labels)
    return selected_object

def get_masks(path, object_name, model, selected_object):
    print('Loading segmentation configuration...')
    masks_start = time.time()
    
    config_path = CONFIG_PATH[model]
    print(config_path)
    model_path = MODEL_PATH[model]
    print(model_path)

    config_path = io.open(config_path, 'r')
    CONFIG = Dict(yaml.load(config_path))
    device = get_device(CUDA)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if CRF else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    masks_path = path + '\\masks'
    create_directory(masks_path)

    index = 1
    imgs_path = path + '\\imgs\\*.png'
    print('Obtaining masks...')

    for img_file in glob.glob(imgs_path):
        img = cv2.imread(img_file)
        height, width = img.shape[:2]
        image, raw_image = preprocessing(img, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        labels = np.unique(labelmap)
        mask = labelmap == selected_object
        mask_final = generate_image_from_mask(mask,width,height)
        cv2.imwrite(masks_path + "\\" + object_name + "_%03d_mask.png" % index, mask_final, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        index += 1
    
    masks_end = time.time()
    masks_time = masks_end - masks_start
    return masks_time

def add_masks(folder, project_name):
    imgs_path = folder + '\\imgs\\*.png'
    masks_path = folder + '\\masks\\*.png'
    dst_path = folder + '\\examples'
    create_directory(dst_path)

    images = []
    masks = []

    for img_file in glob.glob(imgs_path):
        images.append(img_file)
    for mask_file in glob.glob(masks_path):
        masks.append(mask_file)
    
    for i in range(len(images)):
        image = cv2.imread(images[i])
        mask = cv2.imread(masks[i])
        result = cv2.addWeighted(image,1,mask,0.7,0)
        result144 = cv2.resize(result, (256, 144), interpolation = cv2.INTER_AREA)
        result360 = cv2.resize(result, (640, 360), interpolation = cv2.INTER_AREA)
        cv2.imwrite(dst_path + "\\" + project_name + "_%03d.png" % i, result)
        cv2.imwrite(dst_path + "\\" + project_name + "_144_%03d.png" % i, result144)
        cv2.imwrite(dst_path + "\\" + project_name + "_360_%03d.png" % i, result360)

def subtract_background(project_name):
    folder = SETS_PATH + '\\' + project_name
    imgs_path = folder + '\\imgs\\*.png'
    masks_path = folder + '\\masks\\*.png'
    dst_path = folder + '\\without_background'
    create_directory(dst_path)

    images = []
    masks = []

    for img_file in glob.glob(imgs_path):
        images.append(img_file)
    for mask_file in glob.glob(masks_path):
        masks.append(mask_file)
    
    bg_subtraction_start = time.time()
    for i in range(len(images)):
        image = cv2.imread(images[i])
        mask = cv2.imread(masks[i])
        result = cv2.bitwise_and(image,mask)
        cv2.imwrite(dst_path + "\\" + project_name + "_nobg_%03d.png" % i, result)
    bg_subtraction_end = time.time()
    bg_subtraction_time = bg_subtraction_end - bg_subtraction_start
    return bg_subtraction_time


############################
##     IMAGE SAMPLING     ##
############################

def extract_frames(video_path, number_frames_to_extract, object_name, model):
    object_path = SETS_PATH + '\\' + object_name
    imgs_path = object_path + '\\imgs'
    print('- Object path:', object_path)
    create_directory(object_path)
    create_directory(imgs_path)
    
    cap = cv2.VideoCapture(video_path)

    input_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = floor(input_video_frames / number_frames_to_extract)
    count = 1
    index = 1
   
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    while(cap.isOpened() or number_frames_to_extract < index):
        ret, frame = cap.read()
        if ret == True:
            if count % step == 0:
                if index == 1:
                    selected_object = get_label(frame, model)
                    print("Extracting frames...")
                    sampling_start = time.time()
                cv2.imwrite(imgs_path + "\\" + object_name + "_%03d.png" % index, frame)
                index += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            count += 1
        else:
            break
    
    cap.release()

    cv2.destroyAllWindows()

    print("Frame sampling finished.")
    sampling_end = time.time()
    sampling_time = sampling_end - sampling_start
    return object_path, sampling_time, selected_object


############################
##        PHOTOSCAN       ##
############################

def run_photoscan(project_name, project_path, parameters):
    photoscan_start = time.time()
    os.startfile(LICENSE_EXE)
    photoscan_arguments = project_name + ' ' + project_path + ' ' + '"' + parameters + '"'
    cmd_photoscan_script = '"' + PHOTOSCAN_EXE + '" -r "' + PHOTOSCAN_SCRIPT_PATH + '" ' + photoscan_arguments
    print('Photoscan run command string: ' + cmd_photoscan_script)
    subprocess.call(cmd_photoscan_script, shell=True)
    photoscan_end = time.time()
    photoscan_time = photoscan_end - photoscan_start
    return photoscan_time

def get_photoscan_parameters():
    accuracy = 0
    quality = 0
    depth_filtering = 0
    masks = -1
    print('- Photo alignment accuracy')
    print('\t1. Lowest\n\t2. Low\n\t3. Medium\n\t4. High\n\t5. Highest')
    while (accuracy != '1' and accuracy != '2' and accuracy != '3' and accuracy != '4' and accuracy != '5'):
        accuracy = input('\t>> Select an option (1-5): ')
    
    print('- Dense point cloud quality')
    print('\t1. Lowest\n\t2. Low\n\t3. Medium\n\t4. High\n\t5. Ultra High')
    while (quality != '1' and quality != '2' and quality != '3' and quality != '4' and quality != '5'):
        quality = input('\t>> Select an option (1-5): ')
    
    print('- Depth filtering:')
    print('\t1. Disabled\n\t2. Mild\n\t3. Moderate\n\t4. Aggressive')
    while (depth_filtering != '1' and depth_filtering != '2' and depth_filtering != '3' and depth_filtering != '4'):
        depth_filtering = input('\t>> Select an option (1-4): ')

    print('- Do you want to import masks?')
    print('\t0. No\n\t1. Yes')
    while (masks != '0' and masks != '1'):
        masks = input('\t>> Select an option (0-1): ')

    parameter_string = accuracy + ',' + quality + ',' + depth_filtering + ',' + masks
    return parameter_string



############################
##      MAIN PROCESS      ##
############################

def run(video_name):
    # Setting up directories and variables
    total_start = time.time()
    directory_setup()

    print('Video name:', video_name)

    video_path = VIDEOS_PATH + "\\" + video_name + ".mp4"
    check_video(video_path)
    print('Video path:', video_path)

    
    # Asking user for number of frames to extract
    number_of_frames = int(input('How many frames would you like to extract? '))

    # Asking the user for segmentation model
    segmentation_model = select_model()
    print('- Segmentation model selected:', MODEL_NAME[segmentation_model])

    project_name = MODEL_NAME[segmentation_model] + '_' + str(number_of_frames) + '_' + video_name
    print('- Project name:', project_name)

    photoscan_parameter_string = get_photoscan_parameters()
    # Extracting frames and retrieving extraction time
    project_folder, sampling_time, selected_object = extract_frames(video_path, number_of_frames, project_name, segmentation_model)
    print('Frames sampled in', format_timespan(sampling_time))
    print('- Project folder:', project_folder)
    
    # Obtaining masks
    masks_time = get_masks(project_folder, project_name, segmentation_model, selected_object)
    print('Masks obtained in', format_timespan(masks_time))

    # Subtracting background
    subtraction_time = subtract_background(project_name)
    print('Background subtracted in', format_timespan(subtraction_time))

    photoscan_time = run_photoscan(project_name, project_folder, photoscan_parameter_string)
    print('3D reconstruction completed in', format_timespan(photoscan_time))

    total_end = time.time()
    total_time = total_end - total_start
    print('- Times:')
    print('\tEntire process completed in ', format_timespan(total_time))
    print('\t\t- Frames sampled in', format_timespan(sampling_time))
    print('\t\t- Masks obtained in', format_timespan(masks_time))
    print('\t\t- Background subtracted in', format_timespan(subtraction_time))
    print('\t\t- 3D reconstruction completed in', format_timespan(photoscan_time))

    
    # Only to get sample images of the segmentation model.
    # add_masks(project_folder, project_name)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        video_name = input('Enter video name: ')
    elif len(sys.argv) == 2:
        video_name = sys.argv[1]
    else:
        sys.exit('Bad input arguments')
    run(video_name)