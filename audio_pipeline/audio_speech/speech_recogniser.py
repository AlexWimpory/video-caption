import os
import wave
import json
import audio_pipeline.logging_config as logging_config
from tests.vosk_test import config
from vosk import Model, KaldiRecognizer

"""
Module that runs the Vosk model on the input file, returning words and timings
Model can be found at: https://alphacephei.com/vosk/install
Edited to be object orientated and to produce the information required by the project
"""

logger = logging_config.get_logger(__name__)


class SpeechRecogniser:
    """Simple class that uses Vosk to process a file for speech recognition"""

    def __init__(self):
        """Set the log level and load the Vosk model"""
        cwd = os.path.join(os.path.dirname(__file__), config.vosk_model_dir)
        self.model = Model(cwd)
        logger.info(f'Loaded speech recognition model from {cwd}')

    def process_file(self, file_name):
        """
        Run the Vosk model on the input file
        :param file_name: Input wav or mp3 file
        :return: List of dictionaries containing: confidence, start time, end time and the predicted word
        """
        logger.info(f'Recognising speech for {file_name}')
        wf = wave.open(file_name, "rb")
        # Check to see if the audio file can be read by the Vosk model
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
        # Add to results list
        if len(result['text']) > 0:
            results.extend(result['result'])
        logger.info(f'Processed speech, captured {len(results)} results')
        return results


def get_words(results):
    """
    Join words of the speech recogniser's output into a single sentence
    """
    return ' '.join([result['word'] for result in results])
