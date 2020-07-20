import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc, imread

width = 1920
height = 1080
FPS = 60
seconds = 10
video_filename = 'animation.mp4'

interval_size = 0.5 # .5 second intervals for animation 

still_filename = 'still.png'
matt_filename = 'matt.png'
ryan_filename = 'ryan.png'

still_frame = imread(still_filename)
matt_frame = imread(matt_filename)
ryan_frame = imread(ryan_filename)

fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter(video_filename, fourcc, float(FPS), (width,height))

for i in range(FPS*seconds):
    x = np.random.randint(0,3)
    if x == 0:
        video.write(still_frame)
    elif x == 1:
        video.write(matt_frame)
    elif x == 2:
        video.write(ryan_frame)
    else:
        print("bug")

video.release()
