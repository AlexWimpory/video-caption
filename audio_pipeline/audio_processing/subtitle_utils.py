import os
import tempfile
import pysubs2
from pysubs2 import SSAFile, SSAEvent, SSAStyle, make_time
from audio_pipeline.audio_processing.ffmpeg_processor import run_ffmpeg

"""
Subtitle generation tends to create a lot of very small entries across the video file.
To counter this we need to combine together subtitle entries so that they are retained on
the screen as multiple words.  We also want to ensure that subtitles disappear if they pass a
time period since they appeared.  We also want to be able to combine subtitle files in such
a way that the output from each file can be distinguished and covers the correct time periods
"""


def compress(subs, max_chars=30, max_stretch_time=3, max_oldest_time=10):
    new_subs = SSAFile()
    # Phase 1 based on character count so that we dont overflow the screen
    # Phase 2 if the end of the last subtitle is close to the start of the next we want to stretch out the end
    # Phase 3 is to make sure that the oldest word on the screen has not been there for too long
    char_count = 0
    current_text = ''
    current_event = None
    oldest_start_time = 0
    for sub in subs:
        last_event = current_event
        current_event = SSAEvent()
        current_event.start = sub.start
        current_event.end = sub.end
        char_count += len(sub.text)
        # Check the character count and reset if needed
        if char_count > max_chars:
            current_text = sub.text
            char_count = len(sub.text)
        else:
            current_text = current_text + ' ' + sub.text
        # Check the stretch of subtitles make last one longer
        if last_event and current_event.start - last_event.end < max_stretch_time * 1000:
            last_event.end = current_event.start
        else:
            current_text = sub.text
            char_count = len(sub.text)
        # Make sure that the oldest subtitle on the screen is not too old
        if current_event.start - oldest_start_time > max_oldest_time * 1000:
            current_text = sub.text
            char_count = len(sub.text)
            oldest_start_time = sub.start
        current_event.text = current_text
        new_subs.append(current_event)
    return new_subs


def reprocess_subtitle_file(path, max_chars=30, max_stretch_time=3, max_oldest_start=10):
    """Combine subtitles across a time period"""
    file_name, ext = os.path.splitext(path)
    compressed_subtitle_file = file_name + '.ass'
    subs = pysubs2.load(path)
    compressed_subs = compress(subs, max_chars, max_stretch_time, max_oldest_start)
    compressed_subs.save(compressed_subtitle_file)
    return compressed_subtitle_file, subs[len(subs) - 1].end


def create_empty_video(time_in_seconds):
    temp_file_name = tempfile.mktemp(dir='.', prefix='output_', suffix='.mp4')
    run_ffmpeg(f'ffmpeg -y -t {time_in_seconds} -f lavfi -i color=c=black:s=1024*768 -c:v libx264'
               f' -tune stillimage -pix_fmt yuv420p {temp_file_name}')
    return temp_file_name


def add_subtitles_to_video(video_path, subtitle_path):
    temp_file_name = tempfile.mktemp(dir='.', prefix='output_with_subtitles_', suffix='.mp4')
    run_ffmpeg(f'ffmpeg -i {video_path} -i {subtitle_path} -c copy -c:s mov_text {temp_file_name}')
    return temp_file_name


def burn_subtitles_into_video(video_path, subtitle_path, output_path):
    temp_file_name = tempfile.mktemp(dir=output_path, prefix='output_with_hard_subtitles_', suffix='.mp4')
    # Handle srt files if needed
    if subtitle_path.endswith('.srt.'):
        subtitle_ass_file = subtitle_path.replace(".srt", ".ass")
        run_ffmpeg(f'ffmpeg -y -i {subtitle_path} {subtitle_ass_file}')
    else:
        subtitle_ass_file = subtitle_path
    run_ffmpeg(f'ffmpeg -i {video_path} -vf "ass={subtitle_ass_file}" {temp_file_name}')
    return temp_file_name


def combine_subs(first_subs, second_subs, third_subs, fourth_subs, one_only=False):
    # Only ass files keep styling information properly
    combined_subs = SSAFile()
    combined_subs.styles['top'] = SSAStyle(alignment=8)
    combined_subs.styles['bottom'] = SSAStyle(alignment=2)
    combined_subs.styles['left'] = SSAStyle(alignment=4)
    combined_subs.styles['right'] = SSAStyle(alignment=6)

    for sub in second_subs:
        combined_subs.append(SSAEvent(start=sub.start, end=sub.end, text=sub.text, style='bottom'))
    for sub in first_subs:
        combined_subs.append(SSAEvent(start=sub.start, end=sub.end, text=f'({sub.text})', style='top'))
    for sub in third_subs:
        combined_subs.append(SSAEvent(start=sub.start, end=sub.end, text=f'[{sub.text}]', style='left'))
    for sub in fourth_subs:
        combined_subs.append(SSAEvent(start=sub.start, end=sub.end, text=f'[{sub.text}]', style='right'))
    combined_subs.sort()
    if one_only:
        combined_subs = filter_subs(combined_subs)
    return combined_subs


def filter_subs(combined_subs):
    filtered_subs = SSAFile()
    last_sub = None
    last_top_sub = None
    for sub in combined_subs:
        if sub.style == 'bottom':
            filtered_subs.append(sub)
            if last_top_sub and last_top_sub.end > sub.start:
                last_top_sub.end = sub.start
        elif sub.style == 'top':
            if last_sub and last_sub.end > sub.start:
                sub.start = last_sub.end
            if sub.start < sub.end:
                filtered_subs.append(sub)
                last_top_sub = sub
        else:
            filtered_subs.append(sub)
        last_sub = sub
    filtered_removed_empty_subs = SSAFile()
    filtered_removed_empty_subs.styles = combined_subs.styles
    for sub in filtered_subs:
        if sub.end > sub.start:
            filtered_removed_empty_subs.append(sub)
    return filtered_removed_empty_subs


def combine_subtitle_files(first_file, second_file):
    # By default overlapping subtitles are put on separate lines
    file_name, ext = os.path.splitext(first_file)
    combined_subtitle_file = file_name + '_combined.ass'
    # This is the master subtitle file
    combined_subs = combine_subs(pysubs2.load(first_file), pysubs2.load(second_file))
    combined_subs.save(combined_subtitle_file)
    return combined_subtitle_file


def save_to_subtitle_file(results, audio_file_name, f):
    """Converts the output of the model to the standard subtitle format .srt"""
    subs = save_to_subtitles(results, f)
    ass_file = os.path.splitext(audio_file_name)[0] + '.ass'
    subs.save(ass_file)
    return ass_file


def save_to_subtitles(results, f):
    subs = SSAFile()
    for result in results:
        event = SSAEvent(start=make_time(s=result['start']),
                         end=make_time(s=result['end']), text=f(result))
        subs.append(event)
    return subs


if __name__ == '__main__':
    # new_subtitle_file_1, end_time_1 = reprocess_subtitle_file('test.srt')
    # new_subtitle_file_2, end_time_2 = reprocess_subtitle_file('test2.srt')
    # tmp_file = create_empty_video(max(end_time_1, end_time_2) / 1000)
    # new_subtitle_file_combined = combine_subtitle_files(new_subtitle_file_2, new_subtitle_file_1)
    # subtitle_file = burn_subtitles_into_video(tmp_file, new_subtitle_file_combined)
    subs = pysubs2.load('../../out/test_2.ass')
    filter_subs(subs).save('../../out/test_3.ass')
