import nltk
from nltk import FreqDist, WordNetLemmatizer, pos_tag, RegexpParser, ne_chunk
from nltk.corpus import stopwords
from file_utils import load_object


class NaturalLanguageProcessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
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


if __name__ == '__main__':
    #results = load_object('data/YvonneAkiSawyerr_2020T-480p.data')
    #wrds = [result['word'] for result in results]
    wrds = "Sarah was walking her dog in France the trees are green".split()
    nep = NaturalLanguageProcessor()
    print(nep.get_frequency_distribution(wrds, 10))
    print(nep.get_frequency_distribution(nep.remove_stop_words(wrds), 10))
    print(nep.lemmatize(wrds))
    tgd_wrds = nep.pos_tag(wrds)
    print(tgd_wrds)
    print(nep.ner(tgd_wrds))
