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
        timings = [(0, 0)]
        last_partial = ''
        ##use framerate from the file to turn frames into a timer so that each time step through the loop record how many secons of audio reading
        increment = config.frame_to_read / wf.getframerate()
        time = 0
        word_count = 0
        while True:
            data = wf.readframes(config.frame_to_read)
            # if the data we have read is empty then we are at the end of the file
            if len(data) == 0:
                break
            # we now move time forward based on the data we have read
            time = time + increment
            if rec.AcceptWaveform(data):
                ##this for transcript of the file
                # if we reach here we have accepted the translation of a section of text
                ## at some point vosk decides it has a final answer which we grab
                words = json.loads(rec.Result())['text'].split()
                word_count += len(words)
                results += words
                # finally reset the partials
                last_partial = ''
                ##now big list of words
            else:
                ##use partial results to generate timings
                # we have a partial result so use this to remember timings
                partial = rec.PartialResult()
                words = json.loads(rec.PartialResult())['partial'].split()
                if len(words) > 0 and partial != last_partial:
                    ## fixes problems
                    # we have something to process and the partial is different
                    # so the partial might be longer or shorter or the same but with a different set of words
                    if len(words) + word_count > timings[-1][0]:
                        # this is the normal case where vosk has identified a new set of words
                        timings.append((len(words) + word_count, time))
                    elif len(words) + word_count == timings[-1][0]:
                        ## last word different but time has moved on (not sure if ths can happen)
                        # this seems to mean several things including no new audio
                        timings[-1] == (len(words) + word_count, time)
                    else:
                        # sometimes vosk backs up probably due to the way the file is chopped at n frames
                        del timings[-1]
                last_partial = partial
        ## after finished looping grab final results and add to results to stop stuff from being missed at the end (stored in a buffer)
        results += json.loads(rec.FinalResult())['text'].split()
        # we are missing the last few words from timings so add them here
        timings.append((len(results), time))
        return results, timings
