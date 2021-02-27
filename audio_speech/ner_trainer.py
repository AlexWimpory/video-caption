import random
from spacy.util import minibatch, compounding
from pathlib import Path
import spacy

import config

dog_strs = [
    'The {xxx} is barking',
    'The {xxx} is growling',
    'I have a pet {xxx}',
    'I rescued a {xxx} from the shelter',
    'I went for a walk with my {xxx}',
    'My {xxx} has a loud bark',
    'I wish my {xxx} would stop barking',
    'My {xxx} barks at other dogs',
    'My {xxx} barked at the strangers',
    'I took my {xxx} for a walk',
    'My pet {xxx} likes to bark',
    'My {xxx} is from a rescue'
]

dog_breeds = ['German Shepherd Dog',
              'Poodle',
              'Chihuahua',
              'Golden Retriever',
              'Yorkshire Terrier',
              'Dachshund',
              'Beagle',
              'Boxer',
              'Miniature Schnauzer',
              'Shih Tzu',
              'Bulldog',
              'German Spitz',
              'English Cocker Spaniel',
              'Cavalier King Charles Spaniel',
              'French Bulldog',
              'Pug',
              'Rottweiler',
              'English Setter',
              'Maltese',
              'English Springer Spaniel',
              'German Shorthaired Pointer',
              'Staffordshire Bull Terrier',
              'Border Collie',
              'Shetland Sheepdog',
              'Dobermann',
              'West Highland White Terrier',
              'Bernese Mountain Dog',
              'Great Dane',
              'Brittany Spaniel'
              ]


def process_entities(core_strs, replacements, replace_str, entity_type):
    results = []
    for core_str in core_strs:
        for replacement in replacements:
            start = core_str.index(replace_str)
            core_entity_dict = {'entities': [(start, start + len(replacement), entity_type)]}
            new_entity = (core_str.replace(replace_str, replacement.lower()), core_entity_dict)
            results.append(new_entity)
    return results


def run_training(processed_entities, entity_type):
    # Disable pipeline components you dont need to change
    # Resume training
    # Load pre-existing spacy model
    nlp = spacy.load('en_core_web_sm')
    # Getting the pipeline component
    ner = nlp.get_pipe("ner")
    ner.add_label(entity_type)
    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)

    # List of pipes you want to train
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should remain unaffected in training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes):

        sizes = compounding(1.0, 4.0, 1.001)
        # Training for 30 iterations
        for itn in range(30):
            # shuffle examples before training
            random.shuffle(processed_entities)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(processed_entities, size=sizes)
            # dictionary to store losses
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                # Calling update() over the iteration
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                print("Losses", losses)
    return nlp


def save_model(nlp):
    nlp.to_disk(Path(config.nlp_model_dir))


def test_nlp(nlp, sentence):
    doc = nlp(sentence)
    for ent in doc.ents:
        print(ent.label_, ent.text)


if __name__ == '__main__':
    entities = process_entities(dog_strs, dog_breeds, '{xxx}', 'DOG')
    nlp_model = run_training(entities, 'DOG')
    save_model(nlp_model)

    # # nlp_model = spacy.load(config.nlp_model_dir)
    # nlp_model = spacy.load('en_core_web_sm')
    # test_nlp(nlp_model, 'Jack switch on the air conditioner even though it is the middle of summer')
