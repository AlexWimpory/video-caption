from audio_pipeline import pipeline_config


def filter_processor(pos_results):
    """
    Find types of words using POS tagging
    """
    results = []
    for result in pos_results:
        if result['type'] in pipeline_config.pos_types:
            results.append(result)
    return results
