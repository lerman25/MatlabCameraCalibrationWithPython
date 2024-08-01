import shutil
import matlab.engine
import matlab.engine
from CalibrationUtilities import *
import glob
import os
import cv2
import random
from datetime import datetime

def calibrate_camera_from_folder(img_folder,board_size,square_size,radial_coef = 2,max_images_use = 1000,meta_dir = None):
    """
    Create camera calibration parameteres from images in images folder.
 
    Parameters
    ----------
    img_folder : str
        Path to the images folder
    board_size : tupple (int,int)
        Board size
 
    Returns
    -------
    TBD
    """

    imgs = glob.glob(img_folder+'*.png')
    chessboard_mask = chess_board_check(imgs,board_size)
    initial_imgs = [img for i,img in enumerate(imgs) if chessboard_mask[i]]
    if len(initial_imgs) > max_images_use:
        initial_imgs = random.choices(initial_imgs,k=max_images_use)
    eng = matlab.engine()
    # Run Initial calibration on entire set of images
    calibration = calibrate_camera_from_img_list(eng,initial_imgs,square_size,radial_dist_coef=radial_coef)
    cameraParams,imagesUsed,estimationErrors,chess_images_used,chess_points,worldPoints,mrows,ncols = calibration
    # select best images 
    eng.workspace['CameraParams']=cameraParams
    reprojection_errors = eng.eval('CameraParams.ReprojectionErrors')
    reprojection_errors = permute_matlab_point_cloud(reprojection_errors)
    initial_imgs_reprojection_errors = [get_image_reprojection_error(img) for img in reprojection_errors]
    # use k-means to ensure coverage of FOV, select x best images from each mean
    images_points = permute_matlab_point_cloud(chess_points)
    selected_imgs,s_plot_imgs = kmeans_imgs_selection_Matlab(images_points,k=15,count=3,error_threshold=1.5,errors=initial_imgs_reprojection_errors)
    # use best images to calibrate
    final_chess_points = [images_points[i] for i  in selected_imgs]
    # re-"matlab-fy" the points cloud
    final_chess_points = np.array(final_chess_points)
    final_chess_points = np.moveaxis(final_chess_points,0,2)
    final_chess_points = matlab.double([item for item in final_chess_points])

    camera_calibration = calibrate_from_points(eng,final_chess_points,worldPoints,mrows,ncols,radial_coef= radial_coef)

    return
#


if __name__ == "__main__":
    INPUT = ''
    CHESSBOARD_SIZE = (6,8)
    SQUARE_SIZE = 30 # centimeters

# datetime object containing current date and time
    now = datetime.now()
    NOW_DIR  = now.strftime("%d/%m/%Y %H:%M:%S/")
    os.makedirs(NOW_DIR,exist_ok=True)
    FRAMES_DIR = NOW_DIR+'$temp frames dir$/'
    OUTPUT_DIR = NOW_DIR+'matlab calibration output/'
    META_DATA_DIR = NOW_DIR+'matlab calibration meta data/'
    dirs_to_create = [FRAMES_DIR,OUTPUT_DIR,META_DATA_DIR]
    for dir in dirs_to_create:
        os.makedirs(dir,exist_ok=True)

    if os.path.isdir(INPUT):
        img_folder = INPUT
    elif checkIsVideo(INPUT):
        robustSplitToFrames(INPUT,FRAMES_DIR)
        img_folder = INPUT
    else:
        raise ValueError('INPUT is not a directory or not a comptable video file')
    ret_val = calibrate_camera_from_folder(img_folder,board_size=CHESSBOARD_SIZE)
    os.rmdir(FRAMES_DIR)
    
