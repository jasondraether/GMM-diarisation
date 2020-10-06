import numpy as np
import os
from cv2 import VideoWriter, VideoWriter_fourcc, imread

# Given speaker timestamps: [('class'(string),start_timestamp(float64),end_timestamp(float64)),(),(),...,()],
# generate animation for speaker at timestamp
def generate_animation(speaker_timestamps):
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
    class_frames = []
    for class_id in range(len(classes)):
        class_frame_filename = os.path.join(animation_directory,classes[class_id]+'.png')
        frame = imread(class_frame_filename)
        class_frames.append(frame)

    current_time = 0.0
    for datum in speaker_timestamps:
        speaker = datum[0]
        start_timestamp = datum[1]
        end_timestamp = datum[2]
        n_still_frames = int(FPS*(start_timestamp - current_time))
        n_class_frames = int(FPS*(timestamp-current_time))
        for f in range(n_still_frames):



    sample_rate, podcast = wavfile.read(target_podcast)
    n_samples = podcast.shape[0]
    n_seconds = n_samples//sample_rate
    sample_length = int(interval_length*sample_rate)
    frames_per_prediction = int(interval_length*FPS)

    animation_range_upper = (frames_per_prediction*2)//3
    animation_range_lower = (frames_per_prediction)//3


        for f in range(frames_per_prediction):

            if f > animation_range_lower and f < animation_range_upper:
                video.write(still_frame)
            else:
                video.write(class_frames[prediction])

        timestamp = start/sample_rate
        print("Wrote timestamp {0}".format(timestamp))

        if timestamp >= max_seconds:
            print("Exceeded max seconds")
            break

    video.release()
