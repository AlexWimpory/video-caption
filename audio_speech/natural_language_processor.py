import nltk
import spacy
from nltk import FreqDist, WordNetLemmatizer, pos_tag, RegexpParser, ne_chunk
from nltk.corpus import stopwords


class NLTKNaturalLanguageProcessor:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        self._stop_words = stopwords.words('english')
        self._lemma = WordNetLemmatizer()
        self._parser = RegexpParser('NP : {<DT>?<JJ>*<NN>} ')

    @staticmethod
    def get_frequency_distribution(words, num_common):
        return FreqDist(words).most_common(num_common)

    @staticmethod
    def pos_tag(words):
        return pos_tag(words)

    def remove_stop_words(self, words):
        return [word for word in words if word not in self._stop_words]

    def lemmatize(self, words):
        return [self._lemma.lemmatize(word) for word in words]

    def ner(self, tagged_words):
        self._parser.parse(tagged_words)
        return ne_chunk(tagged_words, binary=False)

    @staticmethod
    def process_speech_results(speech_results):
        nlp_results = []
        tagged_words = pos_tag([result['word'] for result in speech_results])
        for i, (word, typ) in enumerate(tagged_words):
            if typ == 'NN':
                speech_result = speech_results[i]
                nlp_results.append({'word': f'{typ}: {word}', 'start': speech_result['start'],
                                    'end': speech_result['start'] + 1, 'conf': speech_result['conf']})
        return nlp_results


class SpaCyResultsProcessor:
    def __init__(self, sentence, natural_language_processor):
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
            end = speech_result['start'] + max(speech_result['end'] - speech_result['start'], 2)
            pos_tag_results.append({'word': word, 'type': typ, 'start': speech_result['start'],
                                    'end': end, 'conf': speech_result['conf']})
        return pos_tag_results

    def process_speech_results_ner(self, speech_results):
        ner_results = []
        nlp_results = self.ner()
        for entity in nlp_results:
            speech_result_start = speech_results[entity[2]]
            speech_result_end = speech_results[entity[3]]
            end = speech_result_start['start'] + max(speech_result_end['end'] - speech_result_start['start'], 2)
            ner_results.append({'word': f'{entity[0]}: {entity[1]}', 'start': speech_result_start['start'],
                                'end': end, 'conf': 1})
        return ner_results


class SpaCyNaturalLanguageProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def get_spacy_results_processor(self, sentence):
        return SpaCyResultsProcessor(sentence, self.nlp)


if __name__ == '__main__':
    # wrds = "Sarah Was Walking Her Dog In France The Trees Are Green".split()
    # nep = NLTKNaturalLanguageProcessor()
    # print(nep.get_frequency_distribution(wrds, 10))
    # print(nep.get_frequency_distribution(nep.remove_stop_words(wrds), 10))
    # print(nep.lemmatize(wrds))
    # tgd_wrds = nep.pos_tag(wrds)
    # print(tgd_wrds)
    # print(nep.ner(tgd_wrds))
    wrds = "aw"
    nlp = SpaCyNaturalLanguageProcessor()
    processor = nlp.get_spacy_results_processor(wrds)
    print(processor.pos_tag())
    print(processor.ner())

