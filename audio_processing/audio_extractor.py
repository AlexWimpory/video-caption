import subprocess
import os
from file_utils import apply_to_path


def extract_audio(file_name, audio_directory):
    """Extract the audio from a .mp4 into a .wav file"""
    basename = os.path.splitext(os.path.basename(file_name))[0]
    audio_file_name = audio_directory + '/' + basename + '.wav'
    subprocess.call(['ffmpeg', '-y', '-i', file_name, '-ac', '1', audio_file_name])
    return audio_file_name


def audio_format(path):
    def f(file_name):
        subprocess.call(['ffprobe', file_name])
    apply_to_path(f, path, '.wav')


if __name__ == '__main__':
    # extract_audio('D:\\TED\\CKWilliams_2001-480p.mp4', 'D:\\TED')
    #audio_format('D:\\Audio Features\\UrbanSound8K\\UrbanSound8K\\audio\\fold5')
    extract_audio('../audio_sounds/data/my_video.mp4', '../audio_sounds/data')