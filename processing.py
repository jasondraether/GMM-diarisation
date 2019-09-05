import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
import wave
import os
from pydub.playback import play
import random

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
    def wav_to_spectrogram(self, samples=None, sample_rate=None, wav_path=None):

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

        print("SPEC LOG SHAPE:")
        print(spectrogram.shape)

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
            if os.path.isdir(wav_path):
                for f in os.listdir(wav_path):
                    sample_rate, samples_read = wavfile.read(wav_path+f)
                    samples += samples_read
            elif os.path.isfile(wav_path):
                sample_rate, samples = wavfile.read(wav_path)
            else:
                print("Unknown data path for .wav files")

        # Check for actual samples
        if len(samples) == 0 or sample_rate == None:
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
                name, ext = os.path.splitext(wav_path)
                filename = 'wav_segments/' + '{:08d}'.format(num_segments) + '.wav'
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
    def label_wav(self, labels=None, filedir=None, file_target='labeled_wavs/'):

        # Check if user gave us any labels
        if labels == None or len(labels) == 0:
            print('Invalid labels provided!')
            return
        else:
            defaults = ['Background','Silence','Unknown','Skip','Replay','Quit']
            lowercase_defaults = [item.lower() for item in defaults]
            lowercase_labels = [item.lower() for item in labels]
            # Remove duplicates
            for label in lowercase_labels:
                if label in lowercase_defaults:
                    removed_index = lowercase_labels.index(label)
                    print('Duplicate label: ', labels[removed_index])
                    labels.remove(labels[removed_index])
            labels.extend(defaults)
        # Check for file directory
        if filedir == None:
            print('Please specify directory of segmented .wav files!')
            return

        # Warn them about writing to default directory
        if file_target == 'labeled_wavs/':
            print('Warning: Using default write directory of '', file_target,'' !')

        # Used to label file number
        file_counter = []

        # Number of labels given
        label_size = len(labels)

        # Start counter at 0 for each label
        for i in range(0, label_size):
            file_counter.append(0)

        # User prompt
        prompt = '\nPlease select one:\n'

        # Craft custom prompt for user, and check if directories exist
        option_index = 0
        for label in labels:
            prompt += str(option_index)
            prompt += ': '
            prompt += label
            prompt += '\n'
            option_index += 1
            # Add directory if it doesn't exist
            if label != 'Quit' and label != 'Replay':
                if not os.path.exists(file_target + label):
                    os.makedirs(file_target + label)

        # Go through all the .wav files and apply labels as specified by user
        file_batch = os.listdir(filedir)
        #random.shuffle(file_batch)
        for filename in sorted(file_batch):
            if filename.endswith('.wav'):
                sample_rate = 0
                sample_rate, samples = wavfile.read(filedir + filename)
                # If we have a bad .wav file, skip over it
                if sample_rate == 0 or len(samples) == 0:
                    print('Invalid WAVE file at: ', filename)
                    continue
                # Play the audio clip
                song = AudioSegment.from_wav(filedir + filename)
                play(song)

                # Loop until valid input or replay song
                while True:
                    input_index = int(input(prompt))
                    # Valid index
                    if input_index >= 0 and input_index < label_size:
                        # Replay audio clip and repeat
                        if labels[input_index] == 'Replay':
                            play(song)
                            continue
                        else:
                            break
                    print('Invalid index! Try again.')
                # Quit option
                if labels[input_index] == 'Quit':
                    print('Quitting...')
                    return
                # Write WAVE file
                else:
                    curr_label = labels[input_index]
                    while True:
                        file_label = file_target + curr_label + '/' + curr_label + '_' + str(file_counter[input_index]) + '.wav'
                        file_counter[input_index] += 1
                        if not os.path.isfile(file_label):
                            break
                    wavfile.write(file_label, sample_rate, samples)
                    print('Writing labeled file at: ', file_label)
            else:
                print('Invalid file type for: ', filename)

# For testing purposes...
def main():
    audio_proc = AudioProcessor()
    segments, sample_rate = audio_proc.parse_wav_data(interval_length=1.0, wav_path='data/', write_enable=True)
    audio_proc.label_wav(labels=['Ryan-Talking', 'Matt-Talking', 'Both-Talking', 'Matt-Laughing', 'Ryan-Laughing', 'Both-Laughing'], filedir='wav_segments/', file_target='labeled_wavs/')
    #for seg in segments:
    #    sample_frequencies, segment_times, spectrogram = audio_proc.wav_to_spectrogram(seg, sample_rate)
    #    audio_proc.plot_spectrogram(sample_frequencies, segment_times, spectrogram)
    #freq, times, spec = audio_proc.wav_to_spectrogram('test.wav')
    #audio_proc.plot_spectrogram(freq, times, spec)

if __name__ == '__main__':
    main()
