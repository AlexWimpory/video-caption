import logging
from audio_pipeline import config
import audio_pipeline.logging_config as logging_config
from audio_pipeline.audio_features.feature_recogniser import FeatureRecogniser
from audio_pipeline.audio_processing.audio_extractor import extract_audio
from audio_pipeline.audio_processing.subtitle_utils import *
from audio_pipeline.audio_speech.natural_language_processor import NLTKNaturalLanguageProcessor, \
    SpaCyNaturalLanguageProcessor
from audio_pipeline.audio_speech.speech_recogniser import SpeechRecogniser

# TODO Logger
logger = logging.getLogger(__name__)


def main(path):
    logging_config.init()

    audio_file = extract_audio(path, config.audio_target_dir)
    print(f'Processing audio file {audio_file}')

    feature_results = FeatureRecogniser().process_file(audio_file)
    speech_results = SpeechRecogniser().process_file(audio_file)
    pos_results = NLTKNaturalLanguageProcessor().process_speech_results(speech_results)
    ner_results = SpaCyNaturalLanguageProcessor().process_speech_results(speech_results)

    subs_1 = save_to_subtitles(speech_results, lambda feature_result: feature_result['word'])
    subs_1 = compress(subs_1)
    subs_2 = save_to_subtitles(feature_results, lambda feature_result: feature_result['class'])
    subs_3 = save_to_subtitles(pos_results, lambda feature_result: feature_result['word'])
    subs_4 = save_to_subtitles(ner_results, lambda feature_result: feature_result['word'])

    combined_subs = combine_subs(subs_2, subs_1, subs_3, subs_4, one_only=True)
    subtitle_file_name = os.path.splitext(path)[0] + '.ass'
    combined_subs.save(subtitle_file_name)
    burn_subtitles_into_video(path, subtitle_file_name, config.audio_target_dir)


if __name__ == '__main__':
    # main('../out/output_with_mono.mkv')
    main('../out/test_5.mp4')
