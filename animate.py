import numpy as np
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread

# Given speaker timestamps: [('class'(string),start_timestamp(float64),end_timestamp(float64)),(),(),...,()],
# generate animation for speaker at timestamp
def generate_animation(speaker_timestamps, classes):
    target_podcast = '' # Input target path of audio file for diarisations
    animation_directory = 'animation_data/' # Path of animation files
    animation_filename = 'animation.mp4' # Path of output animation (without audio)

    # Video parameters
    video_width = 1920
    video_height = 1080
    FPS = 60
    max_seconds = 100 # Animation will stop after max_seconds seconds
    interval_length = 1.0 # How many second intervals to slice podcast into

    fourcc = VideoWriter_fourcc(*'mp4v')
    video = VideoWriter(animation_filename, fourcc, float(FPS), (video_width,video_height))

    # Still image (i.e., nobody talking)
    still_filename = 'still.png'
    still_frame = imread(os.path.join(animation_directory, animation_filename))

    # Read in image of a single speaker talking
    class_frames = {}
    for class_name in classes:
        class_frame_filename = os.path.join(animation_directory,class_name+'.png')
        frame = imread(class_frame_filename)
        class_frames[class_name] = frame

    current_time = 0.0
    for datum in speaker_timestamps:
        speaker = datum[0]
        start_timestamp = datum[1]
        end_timestamp = datum[2]
        n_still_frames = int(FPS*(start_timestamp - current_time))
        n_class_frames = int(FPS*(timestamp-current_time))
        new_frame = still_frame

        for f in range(n_still_frames):
            video.write(new_frame)
        new_frame = class_frames[speaker]
        for f in range(n_class_frames):
            video.write(new_frame)

        if end_timestamp >= max_seconds:
            print("Exceeded max seconds")
            break

    video.release()
