import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
import wave
import os
from pydub.playback import play

'''
Audio processing code for feature extraction
of audio files for neural network
'''
class AudioProcessor:

    '''
    Description: Convert .wav file to spectrogram

    Inputs: samples (ndarray) -- .wav samples
            sample_rate (ndarray) -- .wav sampling rate
            wav_path (string) -- path to .wav file if reading from file

    Outputs: sample_frequencies (ndarray) -- frequencies in wav file
             segment_times (ndarray) -- times in wav file
             spectrogram (ndarray) -- spectrogram values in wav file
    '''
    def wav_to_spectrogram(self, samples, sample_rate, wav_path=None):

        # Check for .wav path
        if wav_path != None:
            sample_rate, samples = wavfile.read(wav_path)

        # Convert wavfile data to spectrogram
        sample_frequencies, segment_times, spectrogram = signal.spectrogram(samples, sample_rate)

        return sample_frequencies, segment_times, spectrogram

    '''
    Description: Plot spectrogram

    Inputs: frequencies (ndarray) -- frequencies to plot
            times (ndarray) -- times to plot
            spectrogram (ndarray) -- spectrogram values to plot

    Outputs: None
    '''
    def plot_spectrogram(self, frequencies, times, spectrogram):

        # Plot spectrogram
        spectrogram = np.log(spectrogram)
        plt.pcolormesh(times, frequencies, spectrogram)
        plt.imshow(spectrogram)

        # Set axes
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time')
        plt.show()

    '''
    Description: Segment wav data into 'length' second intervals

    Inputs: samples (ndarray) -- array of samples from .wav
            sample_rate (ndarray) -- sampling rate of .wav
            interval_length (double) -- time segment size for samples in seconds
            wav_path (string) -- path to wav file if reading from file
            write_enable (boolean) -- true if write segmented data
    Outputs: sample_segments (ndarray) -- array of arrays of segmented samples

    '''
    def parse_wav_data(self, samples=None, sample_rate=None, interval_length=1.0, wav_path=None, write_enable=False):

        # Check for filepath
        if wav_path != None:
            sample_rate, samples = wavfile.read(wav_path)

        # Check for actual samples
        if len(samples) <= 0 or sample_rate == None:
            print("Error with samples and sampling rate!")
            return None

        # Iterator for writing files
        num_segments = 0

        # Time intervals
        t1 = 0.0
        t2 = interval_length

        # Calculate total length of samples in s
        sample_length = len(samples) / sample_rate

        # Hold the segments of samples
        sample_segments = []

        # Continue through all samples
        while(True):
            # Get lower and upper slice of samples
            lower_slice = t1 * sample_rate
            upper_slice = t2 * sample_rate
            # If we're out of bounds, break
            if t2 >= sample_length:
                break
            # Parse current segment
            current_segment = samples[int(lower_slice):int(upper_slice)]
            # Write to output .wav file
            if write_enable == True:
                filename = 'wav_segments/wav_slice' + str(num_segments) + '.wav'
                num_segments += 1
                wavfile.write(filename, sample_rate, current_segment)
            # Append new segment to list
            sample_segments.append(current_segment)
            # Adjust slices for next iteration
            t1 = t2
            t2 = t1 + interval_length

        return sample_segments, sample_rate

    '''
    Description: Sort by hand the segmented wav files. Writes to labeled folders

    Inputs: filedir (string) -- file directory of the segmented wav files
    Outputs: None

    '''
    def label_wav(self, labels=None, filedir=None, file_target='Labeled_Wavs/'):

        # Check if user gave us any labels
        if labels == None:
            print("Invalid labels provided!")
            return None
        else:
            defaults = ['Background','Silence','Unknown','Replay','Quit']
            labels.extend(defaults)

        # Check for file directory
        if filedir == None:
            print("Please specify directory of segmented .wav files!")
            return None

        # Warn them about writing to default directory
        if file_target == 'Labeled_Wavs/':
            print("Warning: Using default write directory of '", file_target,"' !")

        # Used to label file number
        file_counter = []

        # Number of labels given
        label_size = len(labels)

        # Start counter at 0 for each label
        for i in range(0, label_size):
            file_counter.append(0)

        # User prompt
        prompt = ''

        # Craft custom prompt for user
        option_index = 0
        for label in labels:
            prompt += str(option_index)
            prompt += ': '
            prompt += label
            prompt += '\n'
            option_index += 1

        # Input index of replay option
        replay_index = label_size - 2

        # Go through all the .wav files and apply labels as specified by user
        for filename in os.listdir(filedir):
            if filename.endswith(".wav"):
                sample_rate = 0
                # 'song' is the playable variable for audio
                song = AudioSegment.from_wav(filedir + filename)
                with wave.open(filename, "rb") as wave_file:
                    sample_rate = wave_file.getframerate()
                # If we have a bad .wav file, skip over it
                if sample_rate == 0:
                    print("Invalid sampling rate for file: ", filename)
                    continue
                play(song)
                # Loop until valid input or replay song
                while True:
                    input_index = int(input(prompt))
                    if user_input == replay_index:
                        play(song)
                        continue
                    elif input_index >= 0 and input_index < label_size:
                        break
                    print("Invalid index! Try again.")

            else:
                print("Invalid file type for: ", filename)

        # TODO: Make this be able to use arbitrary labels as pased in.
        # Maybe implement a GUI? Also unlimited replays.
        # Also, choose a better filename

        labels = ['','matt/', 'ryan/', 'both/', 'silence/']
        counter = 0

        for filename in os.listdir(filedir):
            if filename.endswith(".wav"):
                song = AudioSegment.from_wav(filedir + filename)
                sample_rate, samples = wavfile.read(filedir + filename)
                for i in range(0, 3):
                    play(song)
                category = int(input("0: Quit\n1: Matt\n2: Ryan\n3: Both\n4: Silence\n5: Replay\nElse: Garbage\n"))
                if category == 0:
                    print("Quitting...")
                    return
                elif category == 5:
                    for i in range(0, 3):
                        play(song)
                elif category > 4 or category < 1:
                    print("Category number out of bounds!")
                else:
                    file_label = labels[category] + str(category) + '_' + str(counter) + '.wav'
                    print("Writing: ", file_label)
                    wavfile.write(file_label, sample_rate, samples)
                    counter += 1
            else:
                print("Incompatible file: ", filename)



# For testing purposes...
def main():
    audio_proc = AudioProcessor()
    segments, sample_rate = audio_proc.parse_wav_data(interval_length=1.0, wav_path='test1.wav', write_enable=True)
    #audio_proc.label_wav('wav_segments/')
    #for seg in segments:
        #sample_frequencies, segment_times, spectrogram = audio_proc.wav_to_spectrogram(seg, sample_rate)
        #audio_proc.plot_spectrogram(sample_frequencies, segment_times, spectrogram)
    #freq, times, spec = audio_proc.wav_to_spectrogram('test.wav')
    #audio_proc.plot_spectrogram(freq, times, spec)

if __name__ == '__main__':
    main()
