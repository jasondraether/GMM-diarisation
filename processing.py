import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
import wave

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
    def parse_wav_data(self, samples, sample_rate, interval_length, wav_path=None, write_enable=False):
        if interval_length < 0.2:
            print("Interval length too low!")
            return None

        # Check for filepath
        if wav_path != None:
            sample_rate, samples = wavfile.read(wav_path)

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
            t1 = t2 + 1.0
            t2 = t1 + interval_length

        return sample_segments, sample_rate

# For testing purposes...
def main():
    audio_proc = AudioProcessor()
    segments, sample_rate = audio_proc.parse_wav_data(0, 0, 1.0, 'test1.wav', True)
    for seg in segments:
        #sample_frequencies, segment_times, spectrogram = audio_proc.wav_to_spectrogram(seg, sample_rate)
        #audio_proc.plot_spectrogram(sample_frequencies, segment_times, spectrogram)
    #freq, times, spec = audio_proc.wav_to_spectrogram('test.wav')
    #audio_proc.plot_spectrogram(freq, times, spec)

if __name__ == '__main__':
    main()
