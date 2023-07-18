import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip
import os

def make_video(save_path, path_to_images, fps=10, save_gif=False):
    img_array = []
    size = None
    for filename in path_to_images:
        # https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python
        with open(filename, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print('Skipping image, not complete')
        else:
            img = cv2.imread(filename, 1)                
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
    
    if len(img_array) == 0:
        print("Not Saving videos, all images were skipped")
        return 

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_dir = '{}/vid.mp4'.format(save_path)
    out = cv2.VideoWriter(save_dir, fourcc, fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    if save_gif:
        # computationally expensive
        try: 
            videoClip = VideoFileClip(save_dir)
            save_dir_gif = '{}/vid.gif'.format(save_path)
            videoClip.write_gif(save_dir_gif)
            print("saving video at: ", save_dir_gif)
        except Exception as e: 
            print("Couldn't Save gif due to {}".format(e))
    # playVideo(save_dir)

def playVideo(file):
    while True:
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture(file)
        cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video Player", 270, 480)
        while (True):

            ret, frame = cap.read()
            # It should only show the frame when the ret is true
            if ret == True:

                cv2.imshow('frame',frame)
                if cv2.waitKey(1) == 27:
                    # When esc is pressed isclosed is 1
                    isclosed=1
                    break
            else:
                break
        # To break the loop if it is closed manually
        if isclosed:
            break

if __name__ == '__main__':
    import sys
    arg = sys.argv[1]
    playVideo(arg)