from audio_processing.ground_truth_processor import GroundtruthReader
from audio_speech.speech_recogniser import SpeechRecogniser
from levenshtein import levenshtein
from functools import partial
from file_utils import apply_to_path
import os
import csv

speech_recogniser = SpeechRecogniser()


def compare_speech(groundtruth, filename):
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    from_audio = speech_recogniser.process_file(filename).upper()
    gtp = GroundtruthReader(groundtruth)
    from_groundtruth = gtp.lookup_filename(base_filename)
    lev = levenshtein(from_audio, from_groundtruth)
    return lev

def write_speech_results(directory, extenstion, groundtruth):
    with open('data/speech_results.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        gtp = GroundtruthReader(groundtruth)
        def write_results(filename):
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            from_groundtruth = gtp.lookup_filename(base_filename)
            from_audio = speech_recogniser.process_file(filename).upper()
            lev = levenshtein(from_audio, from_groundtruth)
            writer.writerow([base_filename, lev, from_audio, from_groundtruth])
            print(base_filename)
        apply_to_path(write_results, directory, extenstion)

if __name__ == '__main__':
    #compare_speech_groundtruth = partial(compare_speech, '../audio_speech/data/librispeech_groundtruth.csv')
    #apply_to_path(compare_speech_groundtruth, 'D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\198', '.wav')
    write_speech_results('D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\198',
                         '.wav',
                         '../audio_speech/data/librispeech_groundtruth.csv')
