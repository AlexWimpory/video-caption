from audio_processing.ground_truth_processor import GroundtruthReader
from audio_speech.speech_recogniser import SpeechRecogniser
from levenshtein import levenshtein
from functools import partial
from file_utils import apply_to_path
import matplotlib.pyplot as plt
import os
import csv
import time
import add_noise

speech_recogniser = SpeechRecogniser()


def compare_speech(groundtruth, filename):
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    results = speech_recogniser.process_file(filename)
    words = [result['word'] for result in results]
    from_audio = ' '.join(words).upper()
    gtp = GroundtruthReader(groundtruth)
    from_groundtruth = gtp.lookup_filename(base_filename)
    lev = levenshtein(from_audio, from_groundtruth)
    return lev


def write_speech_results(path, extenstion, groundtruth):
    with open('data/speech_results.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        gtp = GroundtruthReader(groundtruth)
        def write_results(filename):
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            from_groundtruth = gtp.lookup_filename(base_filename)
            results = speech_recogniser.process_file(filename)
            words = [result['word'] for result in results]
            from_audio = ' '.join(words).upper()
            lev = levenshtein(from_audio, from_groundtruth)
            writer.writerow([base_filename, lev, from_audio, from_groundtruth])
            print(base_filename)
        if os.path.isdir(path):
            apply_to_path(write_results, path, extenstion)
        elif os.path.isfile(path):
            write_results(path)
        else:
            raise Exception('Invalid path')


def format_speech_csv(filename):
    with open(filename, 'r') as fin, open('data/speech_results_formated.txt', 'w', newline='') as fout:
        reader = csv.reader(fin)
        sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
        for row in sortedlist:
            fout.write(f'{row[0]} {row[1]}\n{row[2]}\n{row[3]}\n')


def compare_noise(groundtruth, filename, snr_start, increment, snr_end):
    with open('data/speech_results_noise.txt', 'w', newline='') as fout:
        writer = csv.writer(fout)
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        gtp = GroundtruthReader(groundtruth)
        from_groundtruth = gtp.lookup_filename(base_filename)
        writer.writerow([None, 0, from_groundtruth])
        while snr_start >= snr_end:
            add_noise.add_awgn(filename, snr_start)
            results = speech_recogniser.process_file('data/speech_noisy_fixed.wav')
            words = [result['word'] for result in results]
            from_audio = ' '.join(words).upper()
            lev = levenshtein(from_audio, from_groundtruth)
            writer.writerow([snr_start, lev, from_audio])
            snr_start -= increment


def plot_noise():
    x = []
    y = []

    with open('data/speech_results_noise.txt', 'r') as csvfile:
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
    plt.savefig('data/snr_vs_levenshtein.png')
    plt.show()


if __name__ == '__main__':
    # #Compare speech test
    # compare_speech_groundtruth = partial(compare_speech, '../audio_speech/data/librispeech_groundtruth.csv')
    # apply_to_path(compare_speech_groundtruth, 'D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\198', '.wav')


    # #Write speech test
    # start = time.time()
    # write_speech_results('D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\198\\19-198-0001.wav',
    #                      '.wav',
    #                      '../audio_speech/data/librispeech_groundtruth.csv')
    # end = time.time()
    # print(end - start)
    #
    # # Format speech file
    # format_speech_csv('data/speech_results.csv')


    # Noise test
    compare_noise('../audio_speech/data/librispeech_joined_groundtruth.csv',
                  'data/joined.wav',
                  snr_start = 30,
                  increment = 1,
                  snr_end = 29)
    plot_noise()