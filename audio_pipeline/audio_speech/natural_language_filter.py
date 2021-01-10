from audio_pipeline import config
from audio_pipeline.config import highlight_types


def filter_processor(pos_results):
    results = []
    for result in pos_results:
        if result['type'] in config.pos_types:
            results.append(result)
    for result in results:
        if (result['type'], result['word'].lower()) in highlight_types:
            result['highlight'] = True
    return results
