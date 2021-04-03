import tempfile
from pysubs2 import SSAFile, SSAStyle, Color, SSAEvent, make_time
from audio_pipeline import logging_config
from audio_pipeline.audio_processing.ffmpeg_processor import run_ffmpeg

logger = logging_config.get_logger(__name__)


def _adjust_for_clashing_subs(combined_subs, working_sub, exclude):
    """
    Helper function for the append code.  Looking for overlapping subtitles and make adjustments
    """
    # If we haven't got a set of subs to check against early return
    if not combined_subs or not exclude:
        return working_sub, None
    second_working_sub = None
    for sub in combined_subs:
        # Standard style exit
        if exclude and sub.style not in exclude:
            continue
        if sub.start <= working_sub.start <= sub.end:
            # Drop the start of the working sub
            working_sub.start = sub.end
        elif working_sub.start <= sub.start <= working_sub.end:
            # Drop the end of the working sub
            if sub.end < working_sub.end:
                # We might need to split the sub
                second_working_sub = working_sub.copy()
                second_working_sub.start = sub.end
                second_working_sub.end = working_sub.end
            working_sub.end = sub.start
    # Check that we now have a sub that has no duration
    if working_sub.start >= working_sub.end:
        working_sub = None
    return working_sub, second_working_sub


def append_subs(combined_subs, new_subs, style=None, formatter=None, exclude=None):
    """
    Append a set of subs to a current set avoiding a clash if needed.  Also allows for styling and formatting
    """
    if exclude is None:
        exclude = []
    new_combined_subs = SSAFile()
    if combined_subs:
        # First add the subs we are keeping
        new_combined_subs.extend(combined_subs)
    for sub in new_subs:
        # Add a style
        if style:
            sub.style = style
        # Perform the formatting
        if formatter:
            sub.text = formatter(sub.text)
        # See if we want to cater for clashes
        sub, second_sub = _adjust_for_clashing_subs(combined_subs, sub, exclude)
        # Prepare results
        if sub:
            new_combined_subs.append(sub)
        if second_sub:
            new_combined_subs.append(second_sub)
    new_combined_subs.sort()
    return new_combined_subs


def flatten_subs(starting_subs, style=None):
    """
    Take some subs and merge them together (adjacent subtitle which are the same)
    """
    new_subs = SSAFile()
    for sub in starting_subs:
        # Standard style exit
        if style and sub.style != style:
            continue
        if not new_subs:
            new_subs.append(sub)
        elif sub.text == new_subs[-1].text and sub.start <= new_subs[-1].end:
            if sub.end > new_subs[-1].end:
                new_subs[-1].end = sub.end
        else:
            new_subs.append(sub)
    # Copy in all the subs we skipped due to styling
    if style:
        for sub in starting_subs:
            if sub.style != style:
                new_subs.append(sub)
    new_subs.sort()
    return new_subs


def merge_subs(starting_subs, tolerance_millis=1000, style=None):
    """
    Take some subs and eliminate any blank spots where they are less than a tolerance (default of 1 second)
    """
    merged_subs = SSAFile()
    for sub in starting_subs:
        if style and sub.style != style:
            continue
        if merged_subs and merged_subs[-1].end + tolerance_millis >= sub.start:
            merged_subs[-1].end = sub.start
        merged_subs.append(sub)
    if style:
        for sub in starting_subs:
            if sub.style != style:
                merged_subs.append(sub)
    merged_subs.sort()
    return merged_subs


def compress_subs(subs, max_chars=30, max_stretch_millis=3000, max_oldest_millis=10000, style=None):
    """
    Mostly for the use of speech subtitles this will take individual words and create a running subtitle
    """
    # Phase 1 based on character count so that we dont overflow the screen
    # Phase 2 is to make sure that the oldest word on the screen has not been there for too long
    # First remove gaps where they exist
    merged_subs = merge_subs(subs, max_stretch_millis, style)
    char_count = 0
    oldest_start_time = 0
    compressed_subs = SSAFile()
    for sub in merged_subs:
        if style and sub.style is not style:
            continue
        char_count += len(sub.text)
        # Check the character count and reset if needed
        if char_count > max_chars:
            char_count = len(sub.text)
            oldest_start_time = sub.start
        # Check if subtitle has been on screen for too long then reset
        elif sub.start - oldest_start_time > max_oldest_millis:
            char_count = len(sub.text)
            oldest_start_time = sub.start
        # If there is a gap in time between subtitles then reset
        elif len(compressed_subs) > 0 and sub.start != compressed_subs[-1].end:
            char_count = len(sub.text)
            oldest_start_time = sub.start
        # Add this sub
        elif len(compressed_subs) > 0:
            sub.text = compressed_subs[-1].text + ' ' + sub.text
            char_count += 1
        compressed_subs.append(sub)
    # Append all the other subs
    if style:
        for sub in merged_subs:
            if sub.style is not style:
                compressed_subs.append(sub)
    compressed_subs.sort()
    return compressed_subs


def remove_tiny_subs(subs, duration_millis=1000, left_millis=2000, right_millis=2000, style=None):
    """
    Remove any subs that are out on their own or too short
    """
    copy_subs = SSAFile()
    new_subs = SSAFile()
    for sub in subs:
        if (style and sub.style is style) or not style:
            copy_subs.append(sub)
    for i, sub in enumerate(copy_subs):
        # if it is longer it goes in
        if sub.duration >= duration_millis:
            new_subs.append(sub)
            continue
        # if its the first one then look right only
        # if its the last one then look left only
        # if its in the middle then look both ways
        if left_millis is None and right_millis is None:
            continue
        if i == 0:
            if copy_subs[i + 1].start - sub.end < right_millis:
                new_subs.append(sub)
        elif i == len(copy_subs) - 1:
            if sub.start - copy_subs[i - 1].end < left_millis:
                new_subs.append(sub)
        elif copy_subs[i + 1].start - sub.end < right_millis or sub.start - copy_subs[i - 1].end < left_millis:
            new_subs.append(sub)
    if style:
        for sub in subs:
            if sub.style is not style:
                new_subs.append(sub)
    new_subs.sort()
    return new_subs


def add_styles(subs, style_list=None):
    """
    Add styles to the subtitle file based on the style strings in each individual subtitle
    """
    if style_list is None:
        style_list = []
    for style in style_list:
        new_style = SSAStyle()
        # Number for position refers to the number on a keypad
        if 'top_left' in style:
            new_style.alignment = 7
        elif 'top_right' in style:
            new_style.alignment = 9
        elif 'bottom_left' in style:
            new_style.alignment = 1
        elif 'bottom_right' in style:
            new_style.alignment = 3
        elif 'left' in style:
            new_style.alignment = 4
        elif 'right' in style:
            new_style.alignment = 6
        elif 'top' in style:
            new_style.alignment = 8
        elif 'bottom' in style:
            new_style.alignment = 2
        # Setting the RGB values for the text
        if 'pred' in style:
            new_style.primarycolor = Color(255, 0, 0, 0)
        elif 'pblue' in style:
            new_style.primarycolor = Color(0, 0, 255, 0)
        elif 'pgreen' in style:
            new_style.primarycolor = Color(0, 255, 0, 0)
        elif 'pwhite' in style:
            new_style.primarycolor = Color(255, 255, 255, 0)
        # Setting the RGB values for the text's background
        if 'bred' in style:
            new_style.backcolor = Color(255, 0, 0, 0)
        elif 'bblue' in style:
            new_style.backcolor = Color(0, 0, 255, 0)
        elif 'bgreen' in style:
            new_style.backcolor = Color(0, 255, 0, 0)
        elif 'bwhite' in style:
            new_style.backcolor = Color(255, 255, 255, 0)
        # Setting different font types
        if 'bold' in style:
            new_style.bold = True
        if 'italic' in style:
            new_style.italic = True
        subs.styles[style] = new_style
    return subs


def save_to_subtitles(results, formatter):
    """
    Save to subtitle file
    :param results: Dictionary containing info and start/end times
    :param formatter: Apply text formating to the subtitle
    :return: New subtitle file
    """
    subs = SSAFile()
    for result in results:
        event = SSAEvent(start=make_time(s=result['start']),
                         end=make_time(s=result['end']), text=formatter(result))
        if 'highlight' in result and result['highlight']:
            event.style = 'red'
        subs.append(event)
    logger.info(f'Processed {len(results)} results to subtitle events')
    return subs


def create_styles(subs):
    """
    Gather text from subtitles and call the subtitle adder
    """
    styles = set()
    for sub in subs:
        styles.add(sub.style)
    add_styles(subs, styles)


def burn_subtitles_into_video(video_path, subtitle_path, output_path):
    """
    Create new video with subtitles burned in
    :param video_path: input video path
    :param subtitle_path: subtitle input path
    :param output_path: video output path
    :return: File name that it has written to
    """
    temp_file_name = tempfile.mktemp(dir=output_path, prefix='output_with_hard_subtitles_', suffix='.mp4')
    # Handle srt files if needed
    if subtitle_path.endswith('.srt.'):
        subtitle_ass_file = subtitle_path.replace(".srt", ".ass")
        run_ffmpeg(f'ffmpeg -y -i {subtitle_path} {subtitle_ass_file}')
    else:
        subtitle_ass_file = subtitle_path
    run_ffmpeg(f'ffmpeg -i {video_path} -vf "ass={subtitle_ass_file}" {temp_file_name}')
    logger.info(f'Burnt subtitles {subtitle_path} to {video_path} stored in {temp_file_name}')
    return temp_file_name
