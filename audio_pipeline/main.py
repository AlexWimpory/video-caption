from audio_pipeline.audio_speech.natural_language_filter import filter_processor
from audio_pipeline.logging_config import get_logger
from audio_pipeline import config
from audio_pipeline.audio_sounds.feature_recogniser import FeatureRecogniser
from audio_pipeline.audio_processing.audio_extractor import extract_audio
from audio_pipeline.audio_processing.subtitle_utils import *
from audio_pipeline.audio_speech.natural_language_processor import SpaCyNaturalLanguageProcessor
from audio_pipeline.audio_speech.speech_recogniser import SpeechRecogniser, get_words

logger = get_logger(__name__)


def main(path):
    logger.info(f'Processing video file {path}')
    audio_file = extract_audio(path, config.audio_target_dir)

    sound_results = FeatureRecogniser().process_file(audio_file)
    speech_results = SpeechRecogniser().process_file(audio_file)

    wrds = get_words(speech_results)
    nlp = SpaCyNaturalLanguageProcessor()
    processor = nlp.get_spacy_results_processor(wrds, speech_results)
    pos_results = processor.process_speech_results_tag()
    match_results = processor.process_speech_results_match()
    filtered_pos_results = filter_processor(pos_results, match_results)
    ner_results = processor.process_speech_results_ner()

    subs_1 = save_to_subtitles(speech_results, lambda speech_result: speech_result['word'])
    subs_1 = compress(subs_1)
    subs_2 = save_to_subtitles(sound_results, lambda sound_result: sound_result['class'])
    # subs_2 = flatten_subs(subs_2)
    subs_3 = save_to_subtitles(filtered_pos_results,
                               lambda pos_result: f'{pos_result["type"]} {pos_result["word"]}')
    subs_4 = save_to_subtitles(ner_results,
                               lambda ner_result: f'{ner_result["type"]} {ner_result["word"]}')

    combined_subs = combine_subs(subs_2, subs_1, subs_3, subs_4, one_only=True)
    subtitle_file_name = os.path.splitext(path)[0] + '.ass'
    combined_subs.save(subtitle_file_name)
    burn_subtitles_into_video(path, subtitle_file_name, config.audio_target_dir)
    logger.info(f'Done processing {audio_file}')


if __name__ == '__main__':
    main('../out/demo_8.mp4')
