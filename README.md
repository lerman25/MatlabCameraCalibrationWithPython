# MatlabStereoCalibrationWithPython
Using Matlab's Python engine for stereo calibration (For those who are unfamiliar with Matlab but prefers it over OpenCV), using checkerboard.
Can be used for videos or images.

Originally made for 3 video cameras in a triangle setting: 

![image](https://github.com/user-attachments/assets/a878b724-a359-406f-bc97-7636730acdd4)

Will calibrate Cam1&Cam3, Cam2&Cam3 and will try to calibrate Cam1&Cam2 if possible.

If a single stereo calibration is requied use XXX file.

Pipeline:
1. Split videos of physical checkerboard calibration  into frames (The videos must be synced - starting and ending in the same "real" time).
2. Detect in which frames the checkerboard is visible.
3. Calibrate each camera individually from valid frames to create camera parameteres.
4. Detect in which frames the checkerboard is visible in both cameras.
5. Stereo calibrate each pair of cameras  using camera parameters.
6. Convert to OpenCV format if required.
