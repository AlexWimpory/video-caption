import subprocess
import os


def extract_audio(file_name, audio_directory):
    """Extract the audio from a .mp4 into a .wav file"""
    basename = os.path.basename(file_name).replace('.mp4', '.wav')
    audio_file_name = audio_directory + '/' + basename
    subprocess.call(['ffmpeg', '-y', '-i', file_name, '-ac', '1', audio_file_name])
    return audio_file_name


if __name__ == '__main__':
    extract_audio('D:\\TED\\SheenaIyengar_2010G-480p.mp4', 'D:\\TED')
