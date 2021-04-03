import os
from audio_pipeline import logging_config
from audio_pipeline.audio_processing.ffmpeg_processor import run_ffmpeg

logger = logging_config.get_logger(__name__)


def extract_audio(file_name, audio_directory):
    """Extract the audio from any video format into a .wav file"""
    basename = os.path.splitext(os.path.basename(file_name))[0]
    audio_file_name = audio_directory + '/' + basename + '.wav'
    logger.info(f'Extracting audio from {file_name}')
    run_ffmpeg(f'ffmpeg -y -i {file_name} -ac 1 {audio_file_name}')
    logger.info(f'Done extracting audio from {file_name} saved to {audio_file_name}')
    return audio_file_name
