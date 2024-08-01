# MatlabStereoCalibrationWithPython
Using Matlab's Python engine for stereo calibration (For those who are unfamiliar with Matlab but prefers it over OpenCV), using checkerboard.
Can be used for videos or images.

Originally made for 3 video cameras in a triangle setting: 
![image](https://github.com/user-attachments/assets/a878b724-a359-406f-bc97-7636730acdd4)
Will calibrate Cam1&Cam3, Cam2&Cam3 and will try to calibrate Cam1&Cam2 if possible.

If a single stereo calibration is requied use XXX file.

Pipeline:
1. Split videos of physical checkerboard calibration  into frames.
2. Calibrate each camera individually from frames to create camera parameteres.
3. Stereo calibrate each pair of cameras  using camera parameters.
4. Convert to OpenCV format if required.
