import os
from sounds.mfcc_creator import mfcc_mean, create_mfcc


def prepare_audio_sound(gtp, filter_label, file_name):
    """Generate an MFCC and look up the labels which are matched together"""
    base_file_name = os.path.splitext(os.path.basename(file_name))[0]
    from_groundtruth = gtp.lookup_filename(base_file_name)
    if filter_label is not None and filter_label not in from_groundtruth:
        print(f'skipping {file_name}')
        return None
    mfcc = mfcc_mean(create_mfcc(file_name))
    if filter_label is None:
        return {'mfcc': mfcc, 'labels': from_groundtruth}
    else:
        return {'mfcc': mfcc, 'labels': [filter_label]}
