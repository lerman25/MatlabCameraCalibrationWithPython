import shutil
import matlab.engine
from CalibrationUtilities import *
import os
import cv2
def calibrate_camera_from_img_list():
    return


if __name__ == "__main__":
    INPUT = ''

    FRAMES_DIR = 'c:/$temp frames dir$/'
    if os.path.isdir(INPUT):
        imgs_list = INPUT
    elif checkIsVideo(INPUT):
        robustSplitToFrames(INPUT,FRAMES_DIR)
        imgs_list = INPUT
    else:
        raise ValueError('INPUT is not a directory or not a comptable video file')
    ret_val = calibrate_camera_from_img_list()
    
    