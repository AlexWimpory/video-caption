from audio_speech.speech_recogniser import SpeechRecogniser
from ground_truth.ground_truth_processor import GroundtruthReader
from utils.add_noise import add_awgn, add_real_world_noise
from utils.levenshtein import levenshtein
import matplotlib.pyplot as plt
import os
import csv
import time
from utils.file_utils import apply_to_path

# Initialise the speech recogniser model

speech_recogniser = SpeechRecogniser()

"""
Module which contains several tests for the output of the speech recogniser model
"""


def compare_speech(groundtruth, file_name):
    """Returns the Levenshtein distance when from file name and it's associated ground truth """
    base_file_name = os.path.splitext(os.path.basename(file_name))[0]
    results = speech_recogniser.process_file(file_name)
    words = [result['word'] for result in results]
    from_audio = ' '.join(words).upper()
    gtp = GroundtruthReader(groundtruth)
    from_groundtruth = gtp.lookup_filename(base_file_name)
    lev = levenshtein(from_audio, from_groundtruth)
    print(lev)
    return lev


def write_speech_results(path, extenstion, groundtruth):
    """Creates a .csv file with the calculated Levenstein distance, file name, model output and ground truth
    for each file in the path"""
    with open('../../random_data/speech/data/speech_results.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        gtp = GroundtruthReader(groundtruth)

        def write_results(filename):
            base_file_name = os.path.splitext(os.path.basename(filename))[0]
            from_groundtruth = gtp.lookup_filename(base_file_name)
            results = speech_recogniser.process_file(filename)
            words = [result['word'] for result in results]
            from_audio = ' '.join(words).upper()
            lev = levenshtein(from_audio, from_groundtruth)
            writer.writerow([base_file_name, lev, from_audio, from_groundtruth])
            print(base_file_name)

        if os.path.isdir(path):
            apply_to_path(write_results, path, extenstion)
        elif os.path.isfile(path):
            write_results(path)
        else:
            raise Exception('Invalid path')


def format_speech_csv(filename):
    """Formats the .csv file from write_speech_results to be easier to read and in order"""
    with open(filename, 'r') as fin, open('../../random_data/speech/data/speech_results_formated.txt', 'w', newline='') as fout:
        reader = csv.reader(fin)
        sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
        for row in sortedlist:
            fout.write(f'{row[0]} {row[1]}\n{row[2]}\n{row[3]}\n')


def calculate_accuracy(filename):
    """Use the Levenshtein distance and the total number of characters to find the accuracy"""
    with open(filename, 'r') as fin:
        lev_total = 0
        character_count = 0
        reader = csv.reader(fin)
        for row in reader:
            lev_total += int(float(row[1]))
            character_count += len(row[3])
        error = (lev_total / character_count) * 100
        accuracy = round(100 - error, 2)
        print('The total Levenshtein distance is: ', lev_total)
        print('The ground truth character count is: ', character_count)
        print('The accuracy is: ', accuracy, '%')


def compare_noise(groundtruth, filename, snr_start, increment, snr_end):
    """Adds AWGN to a file and records the model output and Levenshtein distance for each value of SNR"""
    with open('speech_results_noise_2.txt', 'w', newline='') as fout:
        writer = csv.writer(fout)
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        gtp = GroundtruthReader(groundtruth)
        from_groundtruth = gtp.lookup_filename(base_filename)
        writer.writerow([None, 0, from_groundtruth])
        while snr_start >= snr_end:
            add_awgn(filename, snr_start)
            results = speech_recogniser.process_file('speech_noisy_fixed.wav')
            words = [result['word'] for result in results]
            from_audio = ' '.join(words).upper()
            lev = levenshtein(from_audio, from_groundtruth)
            writer.writerow([snr_start, lev, from_audio])
            snr_start -= increment


def plot_snr_vs_lev():
    """Plots a grpah showing the SNR vs the Leveshtein distance"""
    x = []
    y = []
    with open('speech_results_noise.txt', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(csvfile)
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[1]))
    plt.plot(x, y, marker='o', markersize=3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Levenshtein Distance')
    plt.title('SNR vs Levenshtein Distance')
    plt.grid()
    plt.savefig('snr_vs_levenshtein.png')
    plt.show()


if __name__ == '__main__':
    # Compare speech test
    # compare_speech_groundtruth = partial(compare_speech, '../audio_speech/data/librispeech_groundtruth.csv')
    # apply_to_path(compare_speech_groundtruth, 'D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\198', '.wav')

    # # Write speech test
    # start = time.time()
    # # write_speech_results('D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\198',
    # #                      '.wav',
    # #                      '../audio_speech/data/librispeech_groundtruth.csv')
    # write_speech_results('D:\\TED\\CKWilliams_2001-480p.wav',
    #                      '.wav',
    #                      'D:\\TED\\CKWilliams_2001-480p-Transcipt_groundtruth.csv')
    # end = time.time()
    # print(end - start)
    #
    # # Format speech file
    # format_speech_csv('../../random_data/speech/data/speech_results.csv')
    # calculate_accuracy('../../random_data/speech/data/speech_results.csv')

    # # Noise test
    # compare_noise('joined_groundtruth.csv',
    #               'joined.wav',
    #               snr_start = 25,
    #               increment = 1,
    #               snr_end = -25)
    plot_snr_vs_lev()
