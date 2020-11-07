from time import sleep
import pysubs2
from pysubs2 import SSAEvent, make_time, SSAFile
import subprocess

import config


def print_to_time(words, timings):
    ## print text out at the right time
    """Method that outputs the results as a timed string"""
    last_word = 0
    last_time = 0
    char_count = 0
    for time in timings:
        current_string = ' '.join(words[last_word: time[0]])
        char_count += len(current_string)
        if char_count > config.max_print_length:
            print('')
            char_count = 0
        print(current_string, end=' ')
        last_word = time[0]
        # sleep for the time determined in the analysis
        ## sleep for time inbetween words (1.25-0,1.75-1.25 etc)
        sleep(time[1] - last_time)
        last_time = time[1]


def save_to_srt(words, timings, file_name):
    last_time = 0
    last_word = 0
    subs = SSAFile()
    for time in timings:
        text = ' '.join(words[last_word:time[0]])
        event = SSAEvent(start=make_time(s=last_time), end=make_time(s=time[1]), text=text)
        subs.append(event)
        last_time = time[1]
        last_word = time[0]
    subs.save(file_name)


def srt_to_video(file_name_wav, file_name_srt, file_name_image, file_name_video):
    # subprocess.call(['ffmpeg', '-i', file_name_wav, '-loop', '1', '-i', file_name_image, '-c:v',
    # 'libx264', '-r', '24000/1001' , '-vf', f'subtitles={file_name_srt}', '-pix_fmt',
    # 'yuv420p', '-c:a' 'aac', '-map', '1:v', '-map', '0:a', '-shortest', file_name_video])
    subprocess.call(f"ffmpeg -y -i \"{file_name_wav}\" -loop 1 -i {file_name_image}"
                    f" -c:v libx264 -r 24 -vf \"subtitles='{file_name_srt}'\" -pix_fmt yuv420p -c:a aac -map"
                    f" 1:v -map 0:a -shortest {file_name_video}", shell=True)

