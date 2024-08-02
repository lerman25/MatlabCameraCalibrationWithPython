import shutil
import matlab.engine
import matlab.engine
from CalibrationUtilities import *
import glob
import os
import cv2
import random
from datetime import datetime

def calibrate_camera_from_folder(img_folder,board_size,square_size,radial_coef = 2,max_images_use = 1000,meta_dir = None,ret_engine = False):
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
    eng = matlab.engine.start_matlab()
    # Run Initial calibration on entire set of images
    initial_calibration = calibrate_camera_from_img_list(eng,initial_imgs,square_size,radial_dist_coef=radial_coef)
    cameraParams,imagesUsed,estimationErrors,chess_images_used,chess_points,worldPoints,mrows,ncols = initial_calibration
    # select best images 
    eng.workspace['CameraParams']=cameraParams
    reprojection_errors = eng.eval('CameraParams.ReprojectionErrors')
    reprojection_errors = permute_matlab_point_cloud(reprojection_errors)
    initial_imgs_reprojection_errors = [get_image_reprojection_error(img) for img in reprojection_errors]
    # use k-means to ensure coverage of FOV, select x best images from each mean
    images_points = permute_matlab_point_cloud(chess_points)
    img = cv2.imread(initial_imgs[0])
    selected_imgs,kmeans_plot = kmeans_imgs_selection_Matlab(images_points,k=15,count=3,error_threshold=1.5,errors=initial_imgs_reprojection_errors,background_img=img)
    # use best images to calibrate
    final_chess_points = [images_points[i] for i  in selected_imgs]
    # re-"matlab-fy" the points cloud
    final_chess_points = np.array(final_chess_points)
    final_chess_points = np.moveaxis(final_chess_points,0,2)
    final_chess_points = matlab.double([item for item in final_chess_points])

    # now save all the data and maybe convert it to opencv
    camera_calibration = calibrate_from_points(eng,final_chess_points,worldPoints,mrows,ncols,radial_coef= radial_coef)
    cameraCoeff, imagesUsed, estimationErrors = camera_calibration
    
    eng.workspace['CameraParams'] = cameraCoeff
    reprojection_errors = eng.eval('CameraParams.ReprojectionErrors')
    reprojection_errors = permute_matlab_point_cloud(reprojection_errors)
    final_rep_error = np.array([get_image_reprojection_error(img) for img in reprojection_errors]).mean()
    
    final_imgs = [initial_imgs[i] for i in selected_imgs]
    

    if meta_dir:
        eng.workspace['initial_calibration'] = initial_calibration
        eng.save(meta_dir+'initial_calibration.mat', 'initial_calibration',nargout = 0)
        eng.workspace['final_calibration'] = camera_calibration
        eng.save(meta_dir+'final_calibration.mat', 'final_calibration',nargout = 0)
        cv2.imwrite(meta_dir+'camera_kmeans_selection.png',kmeans_plot)
        final_imgs_txt_file = meta_dir+'selected images for final calibration list.txt'
        with open(final_imgs_txt_file,'w') as f:
            for img in final_imgs:
                f.write(img+'\n')
        reprojection_error_txt_file = meta_dir+'calibration reprojection error.txt'
        with open(reprojection_error_txt_file,'w') as f:
            f.write(np.round(final_rep_error,3).astype('str'))

    if ret_engine:
        return camera_calibration,eng
    return camera_calibration
#

import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    INPUT = args[0]
    if len(args)>1:
        os.makedirs(args[1],exist_ok=True)
        os.chdir(args[1]) # for debugging
    CHESSBOARD_SIZE = (6,8)
    SQUARE_SIZE = 30 # centimeters
    now = datetime.now()
    NOW_DIR  = now.strftime("%d-%m-%Y %H-%M-%S/")
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
    
    cameraParams,eng = calibrate_camera_from_folder(img_folder,board_size=CHESSBOARD_SIZE,square_size=SQUARE_SIZE,meta_dir=META_DATA_DIR,ret_engine=True)
    intrinsicMatrix,distortionCoefficients = camera_parameters_to_opencv(eng,cameraParams['final_calibration'][0])
    print(intrinsicMatrix,distortionCoefficients)
    intrinsicMatrix_txt_file = OUTPUT_DIR + 'intrinsicMatrix.txt'
    distortionCoefficients_txt_file = OUTPUT_DIR + 'distortionCoefficients.txt'
    np.savetxt(intrinsicMatrix_txt_file,intrinsicMatrix)
    np.savetxt(distortionCoefficients_txt_file,distortionCoefficients)
    
    if checkIsVideo(INPUT):
        os.rmdir(FRAMES_DIR)
    
