import spacy
import config
from audio_trainers.ner.ner_trainer import process_entities, run_training, save_model, test_nlp
from audio_trainers.ner.training_data import dog_strs, dog_breeds


def train_ner(model=None):
    entities = process_entities(dog_strs, dog_breeds, '{xxx}', 'DOG')
    nlp_model = run_training(entities, 'DOG', model)
    save_model(nlp_model)


def test_ner():
    nlp_model = spacy.load(config.nlp_model_dir)
    test_nlp(nlp_model, 'jack switch on the air conditioner even though it is loud its the middle of summer')
    test_nlp(nlp_model, 'my corgi has a loud bark')
    test_nlp(nlp_model, 'my Poodle barks very loud')
    test_nlp(nlp_model, 'my corgi barks very loud')
    test_nlp(nlp_model, 'Back from red, ferric deserts no Earthly boot had ever touched before')
    test_nlp(nlp_model, 'the rain in spain falls mainly on the plain')


if __name__ == '__main__':
    train_ner()
    test_ner()
