from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json


class SpeechRecogniser:

    def __init__(self):
        SetLogLevel(0)
        self.model = Model("model")

    def process_file(self, file_name):
        wf = wave.open(file_name, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise Exception(f'Invalid file format for {file_name}')
        rec = KaldiRecognizer(self.model, wf.getframerate())
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(json.loads(rec.Result())['text'])
        results.append(json.loads(rec.FinalResult())['text'])
        return ' '.join(results)


if __name__ == '__main__':
    print(SpeechRecogniser().process_file('data/Welcome.wav'))
