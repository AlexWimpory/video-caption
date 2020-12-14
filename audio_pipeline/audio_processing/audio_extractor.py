import subprocess
import os


def extract_audio(file_name, audio_directory):
    """Extract the audio from a .mp4 into a .wav file"""
    basename = os.path.splitext(os.path.basename(file_name))[0]
    audio_file_name = audio_directory + '/' + basename + '.wav'
    subprocess.call(['ffmpeg', '-y', '-i', file_name, '-ac', '1', audio_file_name])
    return audio_file_name
