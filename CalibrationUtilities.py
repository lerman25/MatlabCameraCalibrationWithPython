import cv2
import tqdm
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