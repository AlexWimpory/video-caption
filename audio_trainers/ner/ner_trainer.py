import random
from spacy.util import minibatch, compounding
from pathlib import Path
import spacy
from ner import ner_config


def process_entities(core_strs, replacements, replace_str, entity_type):
    results = []
    for core_str in core_strs:
        for replacement in replacements:
            if replace_str in core_str:
                start = core_str.index(replace_str)
                core_entity_dict = {'entities': [(start, start + len(replacement), entity_type)]}
            else:
                core_entity_dict = {'entities': []}
            new_entity = (core_str.replace(replace_str, replacement.lower()), core_entity_dict)
            results.append(new_entity)
    return results


def run_training(processed_entities, entity_type, model=None):
    # Load model or create a blank model
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')
        print("Created blank 'en' model")

    # Create a NER pipe or find an already existing one
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')
    # Add new entity
    ner.add_label(entity_type)

    # Create an optimiser
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # List of pipes to be trained
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should not be trained
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Disable pipes which should not be trained
    with nlp.disable_pipes(*other_pipes):

        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(ner_config.ner_epochs):
            # Shuffle examples before training
            random.shuffle(processed_entities)
            # Create batches using SpaCy's minibatch
            batches = minibatch(processed_entities, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                # Calling update() over the iteration
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print('Losses', losses)
    return nlp


def save_model(nlp):
    """
    Save trained SpaCy model
    """
    nlp.to_disk(Path(ner_config.nlp_model_dir))


def test_nlp(nlp, sentence):
    doc = nlp(sentence)
    for ent in doc.ents:
        print(ent.label_, ent.text)
