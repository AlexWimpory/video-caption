import subprocess
from subprocess import CalledProcessError
from audio_utils.utils.peek_iterator import peek_iter


def capture_build_info(line, lines_iter):
    build_info = [line]
    while not lines_iter.peek().startswith('Input'):
        build_info.append(next(lines_iter))
    return '\n'.join(build_info)


def capture_info(line, lines_iter, first_line=True):
    in_info = list()
    if first_line:
        in_info.append(line)
    while lines_iter.peek().startswith(' '):
        in_info.append(next(lines_iter))
    if not first_line:
        in_info[0] = in_info[0].strip()
    return '\n'.join(in_info)


def run_ffmpeg(command):
    complete_process = subprocess.run(command, shell=True, capture_output=True)
    complete_process.check_returncode()
    lines = complete_process.stderr.decode('utf-8').split('\n')
    lines_iter = peek_iter(lines)
    results = FFMPEGResults()
    results.return_code = complete_process.returncode
    while lines_iter.has_next():
        line = next(lines_iter)
        if line.startswith('ffmpeg version'):
            results.build_info = capture_build_info(line, lines_iter)
        elif line.startswith('Input #'):
            results.input_info.append(capture_info(line, lines_iter))
        elif line.startswith('Output #'):
            results.output_info.append(capture_info(line, lines_iter))
        elif line.startswith('Stream mapping:'):
            results.stream_info.append(capture_info(line, lines_iter, first_line=False))
        elif line.startswith('Press'):
            continue
        else:
            results.details.append(line)
    return results


class FFMPEGException(BaseException):
    pass


def check_for_error(complete_process):
    """Process the error, throwing a special ffmpeg exception if needed"""
    try:
        complete_process.check_returncode()
    except CalledProcessError as ex:
        lines = complete_process.stderr.decode('utf-8').split('\n')
        if lines[0].startswith('ffmpeg version') or lines[0].startswith('ffprobe version'):
            raise FFMPEGException()
        else:
            raise ex


def run_ffprobe(file_name):
    complete_process = subprocess.run(f'ffprobe "{file_name}"', shell=True, capture_output=True)
    check_for_error(complete_process)
    lines = complete_process.stderr.decode('utf-8').split('\n')
    lines_iter = peek_iter(lines)
    results = FFMPEGResults()
    results.return_code = complete_process.returncode
    while lines_iter.has_next():
        line = next(lines_iter)
        if line.startswith('ffprobe version'):
            results.build_info = capture_build_info(line, lines_iter)
        elif line.startswith('Input #'):
            results.input_info.append(capture_info(line, lines_iter))
        else:
            results.details.append(line)
    return results


class FFMPEGResults:
    def __init__(self):
        self.return_code = 0
        self.build_info = ''
        self.input_info = []
        self.output_info = []
        self.stream_info = []
        self.details = []

    def __str__(self):
        str_list = [f'Return code: {self.return_code}']
        for line in self.input_info:
            str_list.append(line)
        for line in self.output_info:
            str_list.append(line)
        for line in self.stream_info:
            str_list.append(line)
        return '\n'.join(str_list)

    def __repr__(self):
        str_list = [self.build_info]
        for line in self.details:
            str_list.append(line)
        return self.__str__() + '\n' + '\n'.join(str_list)

    # TODO Make this nice
    def get_sample_rate(self):
        return int(self.input_info[0].split('\n')[2].strip().split(',')[1].replace(' Hz', ''))
