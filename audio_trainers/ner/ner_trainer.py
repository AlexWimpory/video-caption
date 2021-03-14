import random
from spacy.util import minibatch, compounding
from pathlib import Path
import spacy
import training_config


def process_entities(core_strs, replacements, replace_str, entity_type):
    results = []
    for core_str in core_strs:
        for replacement in replacements:
            start = core_str.index(replace_str)
            if start != -1:
                core_entity_dict = {'entities': [(start, start + len(replacement), entity_type)]}
            else:
                core_entity_dict = {'entities': []}
            new_entity = (core_str.replace(replace_str, replacement.lower()), core_entity_dict)
            results.append(new_entity)
    return results


def run_training(processed_entities, entity_type, model=None):
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')
        print("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')
    ner.add_label(entity_type)

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # List of pipes you want to train
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # List of pipes which should remain unaffected in training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # Begin training by disabling other pipeline components
    with nlp.disable_pipes(*other_pipes):

        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(training_config.ner_epochs):
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
            print('Losses', losses)
    return nlp


def save_model(nlp):
    nlp.to_disk(Path(training_config.nlp_model_dir))


def test_nlp(nlp, sentence):
    doc = nlp(sentence)
    for ent in doc.ents:
        print(ent.label_, ent.text)
