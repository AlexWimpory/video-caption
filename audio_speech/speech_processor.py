""" Main module for speech processing"""
from speech_recogniser import SpeechRecogniser
from speech_writer import print_to_time, save_to_srt, srt_to_video
import simpleaudio as sa
import os

if __name__ == '__main__':
    f_name = 'D:/Audio Speech/LibriSpeech/train-clean-100/19/198/19-198-0024.wav'
    base_name = os.path.basename(f_name)
    print(f'Processing file {f_name}')
    results = SpeechRecogniser().process_file(f_name)
    print(f'Done processing file {f_name} beginning playback and subtitles')
    print(results)
    print('Starting\n')
    wave_obj = sa.WaveObject.from_wave_file(f_name)
    play_obj = wave_obj.play()
    print_to_time(results)
    save_to_srt(results, base_name.replace('.wav', '.srt'))
    srt_to_video(f_name, base_name.replace('.wav', '.srt'), 'data/blue.png', base_name.replace('.wav', '.mp4'))
    play_obj.wait_done()
    print('\n\nFinished')