# MatlabCameraCalibrationWithPython
%%% The project is still in works (the entire code is written, but not yet organized) 

Using [Matlab's Python engine](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) for stereo/camera calibration (For those who are unfamiliar with Matlab but prefers its camera calibration algorithm over OpenCV's), using chessboard.
Can be used for videos or images.

OpenCV is used in this project for:
1. Chessboard detection (checkChessboard())
2. Split video to frames if needed

Originally made for 3 video cameras in a triangle setting: 

![image](https://github.com/user-attachments/assets/a878b724-a359-406f-bc97-7636730acdd4)

Will calibrate Cam1&Cam3, Cam2&Cam3 and will try to calibrate Cam1&Cam2 if possible.

Process explanation: 
When calibrating from videos frames, one might encouter 3 problems:
  1. Many frames *without* a chessboard visible
  2. Many frames *with* chessboard visible
  3. Too many frames to manually select the best ones, accounting for both FOV coverage and reprojection error
Solutions: 
  1) Matlab chessboard detection is slower then OpenCV's, and OpenCV's is just as good.
     For that reason, the code iterates over each frame/images and checks if a chessboard is visible in that image.
  2) When too many frames/images contains a chessboard visible, the code random samples an appropiate amount of images to use.
  3) To select automaticly from a large set of images, does an initial camera calibration of the entire set (a slow process, but helps for a better result).
     The initial calibration gives reprojection errors for the large set of images.
     To select the best ones accouting for both FOV coverage and reprojection error, the code uses K-means algorithm
     To detect from which sub-set of images, that would be each mean given to image (thus accounting for FOV).
     From each mean it select the ones with the lowest reprojection error (thus accounting for reprojection error).
     


If a single stereo calibration is requied use XXX file.

Pipeline for stereo:
1. Split videos of physical checkerboard calibration  into frames (The videos must be synced - starting and ending in the same "real" time).
2. Detect in which frames the chessboard is visible, factoring in reprojection error and coverage of FOV.
3. Calibrate each camera individually from valid frames to create camera parameteres.
4. Detect in which frames the chessboard is visible in both cameras, factoring in the reprojection error and coverage of FOV.
5. Stereo calibrate each pair of cameras  using camera parameters.
6. Convert to OpenCV format if required.
