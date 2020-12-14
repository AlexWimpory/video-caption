from audio_pipeline import config
from audio_pipeline.audio_features.feature_recogniser import FeatureRecogniser
from audio_pipeline.audio_processing.audio_extractor import extract_audio
from audio_pipeline.audio_processing.srt_writer import add_srt_to_video
from audio_pipeline.audio_speech.speech_recogniser import SpeechRecogniser, save_to_srt


def main(path):
    audio_file = extract_audio(path, config.audio_target_dir)
    print(f'Processing audio file {audio_file}')

    # speech_results = SpeechRecogniser().process_file(audio_file)
    # srt_file = save_to_srt(speech_results, audio_file)
    # add_srt_to_video(srt_file, path)

    feature_results = FeatureRecogniser().process_file(audio_file)
    print(feature_results)


if __name__ == '__main__':
    main('../out/output3.mkv')
