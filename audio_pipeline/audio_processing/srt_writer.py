import os
import subprocess


def add_srt_to_video(file_name_srt, file_name_video):
    """Combines the model generated .srt file with the original video file"""
    file_name_output_video = os.path.splitext(file_name_srt)[0] + '_copy' + os.path.splitext(file_name_video)[1]
    subprocess.call(f"ffmpeg -y -i \"{file_name_video}\" -i \"{file_name_srt}\""
                    f" -map 0:v -map 0:a -c copy -map 1 -c:s:0 mov_text -metadata:s:s:0 language=eng"
                    f" \"{file_name_output_video}\"", shell=True)
