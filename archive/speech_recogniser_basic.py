from vosk import Model, KaldiRecognizer, SetLogLevel
import wave

"""
Module that runs the Vosk model on the input file, returning words and timings
Model can be found at: https://alphacephei.com/vosk/install
Code that comes with the model
"""


def vosk_model(address):
    SetLogLevel(2)

    wf = wave.open(address, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        exit(1)

    model = Model("../audio_utils/tests/vosk_test/model")
    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            print(rec.Result())
        else:
            print(rec.PartialResult())

    print(rec.FinalResult())

