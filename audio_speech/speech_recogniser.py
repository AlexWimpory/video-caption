from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json

import config


class SpeechRecogniser:
    """Simple class that uses Vosk to process a file for speech recognition"""

    def __init__(self):
        """Set the log level and load the Vosk model"""
        SetLogLevel(config.vosk_log_level)
        self.model = Model(config.vosk_model_dir)

    def process_file(self, file_name):
        wf = wave.open(file_name, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise Exception(f'Invalid file format for {file_name}')
        rec = KaldiRecognizer(self.model, wf.getframerate())
        results = []
        while True:
            data = wf.readframes(config.frame_to_read)
            # if the data we have read is empty then we are at the end of the file
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                # result can contain an empty text string but no result list
                if len(result['text']) > 0:
                    # if we reach here we have accepted the translation of a section of text
                    results.extend(result['result'])
        result = json.loads(rec.FinalResult())
        if len(result['text']) > 0:
            results.extend(result['result'])
        return results
