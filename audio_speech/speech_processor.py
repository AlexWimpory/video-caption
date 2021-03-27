from file_utils import save_object
from speech_recogniser import SpeechRecogniser
from speech_writer import print_to_time, save_to_srt, srt_to_video_using_audio, add_srt_to_video
from audio_utils.utils.audio_utils import extract_audio
import simpleaudio as sa

"""
Main module for speech processing
"""


def demo_wav():
    play_audio = True
    audio_file_name = 'data/19-198-0001.wav'
    print(f'Processing audio file {audio_file_name}')
    results = SpeechRecogniser().process_file(audio_file_name)
    print(results)
    save_to_srt(results, audio_file_name.replace('.wav', '.srt'))
    srt_to_video_using_audio(audio_file_name, audio_file_name.replace('.wav', '.srt'), 'data/blue.png',
                             audio_file_name.replace('.wav', '.mp4'))
    print(f'Done processing file {audio_file_name} beginning playback and subtitles')
    if play_audio:
        print('Starting\n')
        wave_obj = sa.WaveObject.from_wave_file(audio_file_name)
        play_obj = wave_obj.play()
        print_to_time(results)
        play_obj.wait_done()
        print('\n\nFinished')
    else:
        print('No playback selected')


def demo_mp4():
    play_audio = False
    video_file_name = 'data/YvonneAkiSawyerr_2020T-480p.mp4'
    audio_directory = 'data/audio'
    print(f'Extracting audio from video {video_file_name}')
    audio_file_name = extract_audio(video_file_name, audio_directory)
    print(f'Processing audio file {audio_file_name}')
    results = SpeechRecogniser().process_file(audio_file_name)
    print(results)
    save_object(results, 'data/YvonneAkiSawyerr_2020T-480p.data')
    save_to_srt(results, audio_file_name.replace('.wav', '.srt'))
    add_srt_to_video(audio_file_name.replace('.wav', '.srt'), video_file_name,
                     video_file_name.replace('.mp4', '_new.mp4'))
    print(f'Done processing file {audio_file_name} beginning playback and subtitles')
    if play_audio:
        print('Starting\n')
        wave_obj = sa.WaveObject.from_wave_file(audio_file_name)
        play_obj = wave_obj.play()
        print_to_time(results)
        play_obj.wait_done()
        print('\n\nFinished')
    else:
        print('No playback selected')


if __name__ == '__main__':
    #demo_wav()
     demo_mp4()
