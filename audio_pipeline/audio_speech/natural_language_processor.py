import spacy
from spacy.matcher.phrasematcher import PhraseMatcher
from audio_pipeline import pipeline_config, logging_config

logger = logging_config.get_logger(__name__)


class SpaCyResultsProcessor:
    def __init__(self, sentence, natural_language_processor, speech_results, matcher):
        logger.debug(f'Processing sentence "{sentence}"')
        self.results = natural_language_processor(sentence)
        self.speech_results = self.pre_process_speech_results(speech_results)
        self.matcher = matcher

    @staticmethod
    def pre_process_speech_results(speech_results):
        count = 0
        for speech_result in speech_results:
            ln = len(speech_result['word'])
            speech_result['char_range'] = range(count, count + ln)
            count = count + ln + 1
        return speech_results

    def pos_tag(self):
        pos_results = []
        for word in self.results:
            pos_results.append((word.text, word.tag_, word.idx))
        return pos_results

    def ner(self):
        ner_results = []
        for entity in self.results.ents:
            ner_results.append((entity.label_, entity.text, entity.start_char, entity.end_char))
        return ner_results

    def noun_chunks(self):
        noun_chunk_results = []
        for chunk in self.results.noun_chunks:
            noun_chunk_results.append((chunk.text, chunk.root.head.text, chunk.start_char, chunk.end_char))
        return noun_chunk_results

    def phrase_match(self):
        phrase_results = []
        matches = self.matcher(self.results)
        for match_id, start, end in matches:
            entity = self.results[start:end]
            phrase_results.append((entity.label_, entity.text, entity.start_char, entity.end_char))
        return phrase_results

    def find_speech_result(self, start_char):
        for speech in self.speech_results:
            if start_char in speech['char_range']:
                return speech

    def process_speech_results_tag(self):
        results = []
        pos_results = self.pos_tag()
        for word, typ, start_char in pos_results:
            speech_result = self.find_speech_result(start_char)
            end = speech_result['start'] + max(speech_result['end'] - speech_result['start'], 1)
            results.append({'word': word, 'type': typ, 'start': speech_result['start'],
                            'end': end, 'conf': speech_result['conf']})
        logger.info(f'Returning POS tagging results captured {len(results)} results')
        return results

    def process_speech_results_ner(self):
        results = []
        ner_results = self.ner()
        for entity in ner_results:
            speech_result_start = self.find_speech_result(entity[2])
            speech_result_end = self.find_speech_result(entity[3] - 1)
            end = speech_result_start['start'] + max(speech_result_end['end'] - speech_result_start['start'], 2)
            results.append({'word': entity[1], 'type': entity[0], 'start': speech_result_start['start'],
                            'end': end, 'conf': 1})
        logger.info(f'Returning NER results captured {len(results)} results')
        return results

    def process_speech_results_match(self):
        results = []
        phrase_results = self.phrase_match()
        for entity in phrase_results:
            speech_result_start = self.find_speech_result(entity[2])
            speech_result_end = self.find_speech_result(entity[3] - 1)
            end = speech_result_start['start'] + max(speech_result_end['end'] - speech_result_start['start'], 2)
            results.append({'word': entity[1], 'type': entity[0], 'start': speech_result_start['start'],
                            'end': end, 'conf': 1})
        logger.info(f'Returning phrase match results captured {len(results)} results')
        return results

    def process_speech_results_chunk(self):
        results = []
        chunk_results = self.noun_chunks()
        for chunk in chunk_results:
            speech_result_start = self.find_speech_result(chunk[2])
            speech_result_end = self.find_speech_result(chunk[3] - 1)
            end = speech_result_start['start'] + max(speech_result_end['end'] - speech_result_start['start'], 2)
            results.append({'word': chunk[0], 'head': chunk[1], 'start': speech_result_start['start'],
                            'end': end, 'conf': 1})
        logger.info(f'Returning phrase chunking results captured {len(results)} results')
        return results


class SpaCyNaturalLanguageProcessor:
    def __init__(self):
        self.nlp = spacy.load(pipeline_config.spacy_model)
        self.matcher = PhraseMatcher(self.nlp.vocab)
        patterns = [self.nlp.make_doc(text) for text in pipeline_config.terms]
        self.matcher.add("TerminologyList", None, *patterns)
        logger.info(f'Loaded NLP model {pipeline_config.spacy_model}')

    def get_spacy_results_processor(self, sentence, speech_results):
        return SpaCyResultsProcessor(sentence, self.nlp, speech_results, self.matcher)

    def process_stopwords(self, speech_results, pos_results):
        new_results = []
        for i, result in enumerate(speech_results):
            left_gap = result['start'] - speech_results[i - 1]['end']
            if i < len(speech_results) - 1:
                right_gap = result['end'] - speech_results[i + 1]['start']
            else:
                right_gap = 10
            if left_gap < 2 or right_gap < 2:
                new_results.append(result)
                continue
            if result['word'] in self.nlp.Defaults.stop_words:
                continue
            if self.get_type(result['start'], pos_results) == 'UH':
                continue
            new_results.append(result)
        return new_results

    @staticmethod
    def get_type(start, pos_results):
        for result in pos_results:
            if result['start'] == start:
                return result['type']


if __name__ == '__main__':
    wrds = "Jack switch on the air conditioner even though it is loud it is the middle of summer"
    nlp = SpaCyNaturalLanguageProcessor()
    processor = nlp.get_spacy_results_processor(wrds, [])
    print(processor.pos_tag())
    print(processor.ner())
    print(processor.noun_chunks())
