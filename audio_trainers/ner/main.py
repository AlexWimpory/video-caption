import spacy
from ner import ner_config
from audio_trainers.ner.ner_trainer import process_entities, run_training, save_model, test_nlp
from audio_trainers.ner.ner_data import dog_strs, dog_breeds


def train_ner(model=None):
    entities = process_entities(dog_strs, dog_breeds, '{xxx}', 'DOG')
    nlp_model = run_training(entities, 'DOG', model)
    save_model(nlp_model)


def test_ner():
    nlp_model = spacy.load(ner_config.nlp_model_dir)
    test_nlp(nlp_model, 'jack switch on the air conditioner even though it is loud its the middle of summer')
    test_nlp(nlp_model, 'my corgi has a loud bark')
    test_nlp(nlp_model, 'my Poodle barks very loud')
    test_nlp(nlp_model, 'my corgi barks very loudly')
    test_nlp(nlp_model, 'To be or not to be that is the question')
    test_nlp(nlp_model, 'jack switch on the air conditioner it is the middle of june my dog sam likes to bark at strangers this is a problem when we go to the park')
    test_nlp(nlp_model, 'The big red car drove down the road')
    test_nlp(nlp_model, 'my corgi barks loudly at scary strangers')


if __name__ == '__main__':
    # train_ner()
    test_ner()
