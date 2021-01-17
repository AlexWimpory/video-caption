from audio_pipeline import config


def filter_processor(pos_results, match_results):
    results = []
    for result in pos_results:
        if result['type'] in config.pos_types:
            results.append(result)
    for result in match_results:
        result['highlight'] = True
        results.append(result)
    return results
