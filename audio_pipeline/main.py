from audio_pipeline.logging_config import get_logger
from audio_pipeline import pipeline_config
from audio_pipeline.audio_sounds.sound_recogniser import FeatureRecogniser, process_overlap
from audio_pipeline.audio_processing.audio_extractor import extract_audio
from audio_pipeline.audio_processing.subtitle_utils import *
from audio_pipeline.audio_speech.natural_language_processor import SpaCyNaturalLanguageProcessor
from audio_pipeline.audio_speech.speech_recogniser import SpeechRecogniser, get_words
import os

logger = get_logger(__name__)


def main(path):
    logger.info(f'Processing video file {path}')
    audio_file = extract_audio(path, pipeline_config.audio_target_dir)

    sound_results = FeatureRecogniser().process_file(audio_file)
    sound_results = process_overlap(sound_results)
    speech_results = SpeechRecogniser().process_file(audio_file)

    wrds = get_words(speech_results)
    nlp = SpaCyNaturalLanguageProcessor(pipeline_config.spacy_model)
    custom_nlp = SpaCyNaturalLanguageProcessor(pipeline_config.custom_spacy_model)
    processor = nlp.get_spacy_results_processor(wrds, speech_results)
    custom_processor = custom_nlp.get_spacy_results_processor(wrds, speech_results)
    chunk_results = processor.process_speech_results_chunk()
    ner_results = processor.process_speech_results_ner()
    ner_results.extend(custom_processor.process_speech_results_ner())
    match_results = processor.process_speech_results_match()
    speech_results = nlp.process_stopwords(speech_results, chunk_results)

    subs_1 = save_to_subtitles(speech_results,
                               lambda speech_result: speech_result['word'])
    subs_1 = compress_subs(subs_1)
    subs_2 = save_to_subtitles(sound_results,
                               lambda sound_result: sound_result['class'])
    subs_2 = flatten_subs(subs_2)
    subs_3 = save_to_subtitles(chunk_results,
                               lambda chunk_result: f'{chunk_result["word"]} ({chunk_result["head"]})')
    subs_4 = save_to_subtitles(ner_results,
                               lambda ner_result: f'{ner_result["type"]} {ner_result["word"]}')
    subs_5 = save_to_subtitles(match_results,
                               lambda match_result: match_result["word"])

    combined_subs = append_subs(None, subs_1, style='bottom')
    combined_subs = append_subs(combined_subs, subs_2, exclude=['bottom'], style='top', formatter=lambda x: f'({x})')
    combined_subs = append_subs(combined_subs, subs_3, style='left')
    combined_subs = append_subs(combined_subs, subs_4, style='right')
    combined_subs = append_subs(combined_subs, subs_5, style='bottom_left_pred')
    combined_subs = remove_tiny_subs(combined_subs, duration_millis=1000, left_millis=None,
                                     right_millis=None, style='top')

    subtitle_file_name = os.path.splitext(path)[0] + '.ass'
    create_styles(combined_subs)
    combined_subs.save(subtitle_file_name)
    burn_subtitles_into_video(path, subtitle_file_name, pipeline_config.audio_target_dir)
    logger.info(f'Done processing {audio_file}')


if __name__ == '__main__':
    main('../out/demo_12.mp4')
