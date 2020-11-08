from time import sleep
import pysubs2
from pysubs2 import SSAEvent, make_time, SSAFile
import subprocess

import config


def print_to_time(results):
    """Method that outputs the results as a timed string"""
    last_time = 0
    char_count = 0
    for result in results:
        char_count += len(result['word'])
        if char_count > config.max_print_length:
            print('')
            char_count = 0
        print(result['word'], end=' ')
        # sleep for the time determined in the analysis
        sleep(result['end'] - last_time)
        last_time = result['end']


def save_to_srt(results, file_name):
    subs = SSAFile()
    for result in results:
        event = SSAEvent(start=make_time(s=result['start']), end=make_time(s=result['end']), text=result['word'])
        subs.append(event)
    subs.save(file_name)


def srt_to_video(file_name_wav, file_name_srt, file_name_image, file_name_video):
    subprocess.call(f"ffmpeg -y -i \"{file_name_wav}\" -loop 1 -i {file_name_image}"
                    f" -c:v libx264 -r 24 -vf \"subtitles='{file_name_srt}'\" -pix_fmt yuv420p -c:a aac -map"
                    f" 1:v -map 0:a -shortest {file_name_video}", shell=True)

