from vosk import Model, KaldiRecognizer, SetLogLevel
import wave


def vosk_model(address):
    SetLogLevel(2)

    wf = wave.open(address, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        exit(1)

    model = Model("model")
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


if __name__ == '__main__':
    vosk_model('D:\\Audio Speech\\LibriSpeech\\train-clean-100\\19\\227\\19-227-0005.wav')
