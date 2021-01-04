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


class SpaCyNaturalLanguageProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def ner(self, sentence):
        return self.nlp(sentence)

    def process_speech_results(self, speech_results):
        nlp_results = []
        ner_words = self.nlp(' '.join([result['word'] for result in speech_results]))
        for entity in ner_words.ents:
            nlp_results.append({'word': f'{entity.label_}: {entity.text}', 'start': 0,
                                'end': 10, 'conf': 1})
        return nlp_results


if __name__ == '__main__':
    # wrds = "Sarah Was Walking Her Dog In France The Trees Are Green".split()
    # nep = NLTKNaturalLanguageProcessor()
    # print(nep.get_frequency_distribution(wrds, 10))
    # print(nep.get_frequency_distribution(nep.remove_stop_words(wrds), 10))
    # print(nep.lemmatize(wrds))
    # tgd_wrds = nep.pos_tag(wrds)
    # print(tgd_wrds)
    # print(nep.ner(tgd_wrds))
    wrds = "sarah was walking her dog in france the trees are green"
    nep = SpaCyNaturalLanguageProcessor()
    results = nep.nlp(wrds)
    print(results)
