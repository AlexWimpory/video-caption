import os
from vosk import Model, KaldiRecognizer, SetLogLevel
from pysubs2 import SSAEvent, make_time, SSAFile
import wave
import json
import config

"""
Module that runs the Vosk model on the input file, returning words and timings
Model can be found at: https://alphacephei.com/vosk/install
Edited to be object orientated and to produce the information required by the project
"""


class SpeechRecogniser:
    """Simple class that uses Vosk to process a file for speech recognition"""

    def __init__(self):
        """Set the log level and load the Vosk model"""
        SetLogLevel(config.vosk_log_level)
        cwd = os.path.join(os.path.dirname(__file__), config.vosk_model_dir)
        self.model = Model(cwd)

    def process_file(self, file_name):
        wf = wave.open(file_name, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise Exception(f'Invalid file format for {file_name}')
        rec = KaldiRecognizer(self.model, wf.getframerate())
        results = []
        while True:
            data = wf.readframes(config.frame_to_read)
            # If the data we have read is empty then we are at the end of the file
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                # Result can contain an empty text string but no result list
                if len(result['text']) > 0:
                    # If we reach here we have accepted the translation of a section of text
                    results.extend(result['result'])
        result = json.loads(rec.FinalResult())
        if len(result['text']) > 0:
            results.extend(result['result'])
        return results


def get_words(results):
    return ' '.join([result['word'] for result in results])


def save_to_srt(results, audio_file_name):
    """Converts the output of the model to the standard subtitle format .srt"""
    subs = SSAFile()
    for result in results:
        event = SSAEvent(start=make_time(s=result['start']), end=make_time(s=result['end']), text=result['word'])
        subs.append(event)
    srt_file = os.path.splitext(audio_file_name)[0] + '.srt'
    subs.save(srt_file)
    return srt_file
