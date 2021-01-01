from audio_pipeline import config
from audio_pipeline.audio_features.feature_recogniser import FeatureRecogniser
from audio_pipeline.audio_processing.audio_extractor import extract_audio
from audio_pipeline.audio_processing.subtitle_utils import *
from audio_pipeline.audio_speech.speech_recogniser import SpeechRecogniser


def main(path):
    #audio_file = extract_audio(path, config.audio_target_dir)
    audio_file = '../out/aa.wav'
    print(f'Processing audio file {audio_file}')

    speech_results = SpeechRecogniser().process_file(audio_file)
    subs_1 = save_to_subtitles(speech_results, lambda feature_result: feature_result['word'])
    subs_1 = compress(subs_1)

    feature_results = FeatureRecogniser().process_file(audio_file)
    subs_2 = save_to_subtitles(feature_results, lambda feature_result: feature_result['class'])

    combined_subs = combine_subs(subs_2, subs_1)
    subtitle_file_name = os.path.splitext(path)[0] + '.ass'
    combined_subs.save(subtitle_file_name)
    file_name = create_empty_video(40)
    burn_subtitles_into_video(file_name, subtitle_file_name, config.audio_target_dir)


if __name__ == '__main__':
    # main('../out/output_with_mono.mkv')
    main('../out/zzz.mp4')
