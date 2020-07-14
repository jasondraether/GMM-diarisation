

def label_wav(labels=None, filedir=None, file_target='labeled_wavs/'):
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


def main():
    pass

if __name__ == '__main__':
    main()
