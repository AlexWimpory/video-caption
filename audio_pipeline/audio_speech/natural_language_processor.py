import spacy
from audio_pipeline import config, logging_config

# TODO
# add to ner training
# look at phrase chunking

logger = logging_config.get_logger(__name__)


class SpaCyResultsProcessor:
    def __init__(self, sentence, natural_language_processor):
        logger.debug(f'Processing sentence "{sentence}"')
        self.results = natural_language_processor(sentence)

    def pos_tag(self):
        nlp_results = []
        for word in self.results:
            nlp_results.append((word.text, word.tag_))
        return nlp_results

    def ner(self):
        nlp_results = []
        for entity in self.results.ents:
            nlp_results.append((entity.label_, entity.text, entity.start, entity.end))
        return nlp_results

    def process_speech_results_tag(self, speech_results):
        pos_tag_results = []
        nlp_results = self.pos_tag()
        for i, (word, typ) in enumerate(nlp_results):
            speech_result = speech_results[i]
            end = speech_result['start'] + max(speech_result['end'] - speech_result['start'], 1)
            pos_tag_results.append({'word': word, 'type': typ, 'start': speech_result['start'],
                                    'end': end, 'conf': speech_result['conf']})
        logger.info(f'Returning POS tagging results captured {len(pos_tag_results)} results')
        return pos_tag_results

    def process_speech_results_ner(self, speech_results):
        ner_results = []
        nlp_results = self.ner()
        for entity in nlp_results:
            speech_result_start = speech_results[entity[2]]
            speech_result_end = speech_results[entity[3]]
            end = speech_result_start['start'] + max(speech_result_end['end'] - speech_result_start['start'], 1)
            ner_results.append({'word': f'{entity[0]}: {entity[1]}', 'start': speech_result_start['start'],
                                'end': end, 'conf': 1})
        logger.info(f'Returning NER results captured {len(ner_results)} results')
        return ner_results


class SpaCyNaturalLanguageProcessor:
    def __init__(self):
        self.nlp = spacy.load(config.spacy_model)
        logger.info(f'Loaded NLP model {config.spacy_model}')

    def get_spacy_results_processor(self, sentence):
        return SpaCyResultsProcessor(sentence, self.nlp)


if __name__ == '__main__':
    wrds = "Jack switch on the air conditioner even though it is loud its the middle of summer"
    nlp = SpaCyNaturalLanguageProcessor()
    processor = nlp.get_spacy_results_processor(wrds)
    print(processor.pos_tag())
    print(processor.ner())
