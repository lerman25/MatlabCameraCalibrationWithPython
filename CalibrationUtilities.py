import cv2
import tqdm
import matlab
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import os

def checkIsVideo(file_path):
    cap = cv2.VideoCapture(file_path)
    hasFrame, x = cap.read()
    return hasFrame
def robustSplitToFrames(video_source_path, output_dir_path):
    """
    Split video to frames as images in output_dir_path
    Using timestamps rather then iterative read() - 
    To make sure the amount of output images is the same if another videos with the same length is also spiltted 
    """
    count = 0
    vidcap = cv2.VideoCapture(video_source_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    def getFrame(sec,max_frame = None):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(output_dir_path+str(count).zfill(len(str(max_frame)))+".png", image) # Save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 1/30 # Change this number to 1 for each 1 second
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps
    success = getFrame(sec,max_frame=int(durationInSeconds*30))
    frames_buffer_count = 100
    frames_buffer = []
    buffer_counter = 0
    total_frames = int(durationInSeconds*30)
    total_frames_str = str(total_frames)
    total_frames_len = len(total_frames_str)
    with tqdm.tqdm(total=total_frames) as pbar:
        while success:
            count = count + 1
            if count%10==0:
                pbar.update(10)
            sec = sec + frameRate
            # success = getFrame(sec,max_frame=int(durationInSeconds*30))
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            success,image = vidcap.read()
            if success:
                frames_buffer.append((output_dir_path+str(count).zfill(total_frames_len)+".png",image))
                buffer_counter+=1
                # cv2.imwrite(output_dir_path+str(count).zfill(len(str(int(durationInSeconds*30))))+".png", image)
            if buffer_counter == frames_buffer_count:
            #empty buffer 
                for path,frame in frames_buffer:
                    cv2.imwrite(path,frame)
                buffer_counter=0
                frames_buffer = []
    if buffer_counter>0:
        for path,frame in frames_buffer:
            cv2.imwrite(path,frame)
            buffer_counter=0
            frames_buffer = []
def chess_board_check(img_lst,board_size):
    results = []
    for i,img_path in enumerate(tqdm.tqdm(img_lst)):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if cv2.checkChessboard(img,board_size):
            results.append(True)
        else:   
            results.append(False)
    return results
def calibrate_camera_from_img_list(eng,img_lst,square_Size,radial_dist_coef = 2):
    f_detector = eng.vision.calibration.monocular.CheckerboardDetector()
    ret_val=eng.detectPatternPoints(f_detector,img_lst,nargout=2)
    chess_points = ret_val[0]
    chess_images_used = ret_val[1]
    originalImage = eng.imread(img_lst[0],nargout=1)
    (mrows,ncols,o)= eng.size(originalImage,nargout=3)
    squareSize = square_Size;  # in units of 'centimeters'
    worldPoints = eng.generateWorldPoints(f_detector, 'SquareSize', squareSize,nargout=1)
    (cameraParams3coeff, imagesUsed, estimationErrors) = eng.estimateCameraParameters(chess_points, worldPoints, 
        'EstimateSkew', False, 'EstimateTangentialDistortion', True, 
        'NumRadialDistortionCoefficients', radial_dist_coef, 'WorldUnits', 'centimeters', 
        'InitialIntrinsicMatrix', matlab.double(vector=[]  ), 'InitialRadialDistortion', matlab.double(vector=[]  ),
        'ImageSize', matlab.double(vector=[mrows,ncols]  ),nargout = 3)
    return cameraParams3coeff,imagesUsed,estimationErrors,chess_images_used,chess_points,worldPoints,mrows,ncols
def permute_matlab_point_cloud(points):
    """gets points shape [x,y,z] return [z,x,y]
    example [48,2,107] -> [107,48,2]"""
    points_rot = []
    for o in range(points.size[2]):
        img = []
        for i in range(points.size[0]):
            point = []
            for j in range(points.size[1]):
                point.append(points[i][j][o])
            img.append(point)
        points_rot.append(img)
    return points_rot
def get_image_reprojection_error(img_reprojection_errors):
    errors_sum = 0 
    for point in img_reprojection_errors:
        point_error = (point[0]**2+point[1]**2)**0.5
        errors_sum+=point_error
    return errors_sum/len(img_reprojection_errors)
def kmeans_imgs_selection_Matlab(corners_list,k = 10,errors = None,count = 3,error_threshold = 1.5,recursive_error_fix =False,background_img = None):
    """
    $$ function code needs organzing
    This function tries to find the best FOV coverage from calibration images using k-means algorithm
    From each mean, the function will select the images with the lowest reprojection error
    will ignoring reprojection error that are above certain threshold
    If not enough images are above the threshold, you can use recursive_error_fix that will recursivly call the function
    with higher threshold.
    The function also plots the selected mean to visualize the coverage
    """
    images_first_corners = [corner[0][:] for corner in corners_list]
    corners_first_cord = images_first_corners
    corners_first_cord = np.array(corners_first_cord,dtype=np.double)
    corners_first_cord=np.nan_to_num(corners_first_cord)
    h = 0.5 # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = corners_first_cord[:, 0].min() - 1, corners_first_cord[:, 0].max() + 1
    y_min, y_max = corners_first_cord[:, 1].min() - 1, corners_first_cord[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_s = [cfc[0] for cfc in corners_first_cord]
    y_s = [cfc[1] for cfc in corners_first_cord]
    kmeans = KMeans(n_clusters=k,n_init='auto')
    train_labels = kmeans.fit_predict(corners_first_cord)
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(1)
    plt.clf()
    Z = np.expand_dims(Z, axis=0)
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    if background_img is not None:
        plt.imshow(background_img)

    plt.plot(corners_first_cord[:, 0], corners_first_cord[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on calibration images\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    closest_data = []
    m_clusters = kmeans.labels_.tolist()
    all_data = [ corners_first_cord.shape[0]]

    centers = np.array(kmeans.cluster_centers_)
    if errors:
        closest = []
        labels =train_labels
        array = np.full((k,len(corners_list)),fill_value = np.inf)
        for i,label in enumerate(labels):
            array[label][i] = errors[i]
        argsorted = np.argsort(array)
        for i in range(k):
            for j in range(count):
                if array[i][argsorted[i][j]] < error_threshold:
                    closest.append(argsorted[i][j])
        for clo in closest:
            plt.scatter(
                corners_first_cord[clo][0],
                corners_first_cord[clo][1],
                marker="x",
                s=50,
                linewidths=3,
                color="black",
                zorder=10,
            )
    if not errors:
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, corners_first_cord)
        for clo in closest:
            plt.scatter(
                corners_first_cord[clo][0],
                corners_first_cord[clo][1],
                marker="x",
                s=50,
                linewidths=3,
                color="black",
                zorder=10,
            )

    plt.savefig('--------temp.png')
    plot_img = cv2.imread('--------temp.png',-1)
    os.remove('--------temp.png')
    if len(closest)==0:
        if recursive_error_fix:
            return kmeans_imgs_selection_Matlab(corners_list,k,errors,count,error_threshold+1,True)
    return closest,plot_img
def calibrate_from_points(eng,imagePoints,worldPoints,mrows,ncols,radial_coef = 2):
        (cameraParams3coeff, imagesUsed, estimationErrors) = eng.estimateCameraParameters(imagePoints, worldPoints, 
        'EstimateSkew', False, 'EstimateTangentialDistortion', True, 
        'NumRadialDistortionCoefficients', radial_coef, 'WorldUnits', 'centimeters', 
        'InitialIntrinsicMatrix', matlab.double(vector=[]  ), 'InitialRadialDistortion', matlab.double(vector=[]  ),
        'ImageSize', matlab.double(vector=[mrows,ncols]  ),nargout = 3)
        return (cameraParams3coeff, imagesUsed, estimationErrors)
#
