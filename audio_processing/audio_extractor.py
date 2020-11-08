import subprocess
import os


def extract_audio(file_name, audio_directory):
    basename = os.path.basename(file_name).replace('.mp4', '.wav')
    audio_file_name = audio_directory + '/' + basename
    subprocess.call(['ffmpeg', '-y', '-i', file_name, '-ac', '1', audio_file_name])
    return audio_file_name
