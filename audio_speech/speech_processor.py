""" Main module for speech processing"""
from speech_recogniser import SpeechRecogniser
from speech_writer import print_to_time, save_to_srt, srt_to_video
import simpleaudio as sa
import os

if __name__ == '__main__':
    f_name = 'D:/Audio Speech/LibriSpeech/train-clean-100/19/198/19-198-0024.wav'
    base_name = os.path.basename(f_name)
    print(f'Processing file {f_name}')
    wrds, tims = SpeechRecogniser().process_file(f_name)
    print(f'Done processing file {f_name} beginning playback and subtitles')
    print(wrds)
    print(tims)
    print('Starting\n')
    ##simple audio for playing wav file from python
    wave_obj = sa.WaveObject.from_wave_file(f_name)
    play_obj = wave_obj.play()
    ## Stuff in speech writer
    print_to_time(wrds, tims)
    save_to_srt(wrds, tims, base_name.replace('.wav', '.srt'))
    srt_to_video(f_name, base_name.replace('.wav', '.srt'), 'data/blue.png', base_name.replace('.wav', '.mp4'))
    ##Make sure audio finishes playing before the program exits
    play_obj.wait_done()
    print('\n\nFinished')