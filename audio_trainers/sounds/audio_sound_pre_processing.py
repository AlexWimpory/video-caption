from ground_truth_processor import GroundtruthReader
import os
from sounds.mfcc_creator import mfcc_mean, create_mfcc


def prepare_audio_sound(groundtruth, file_name):
    """Generate an MFCC and look up the labels which are matched together"""
    base_file_name = os.path.splitext(os.path.basename(file_name))[0]
    gtp = GroundtruthReader(groundtruth)
    from_groundtruth = gtp.lookup_filename(base_file_name)
    mfcc = mfcc_mean(create_mfcc(file_name))
    return {'mfcc': mfcc, 'labels': from_groundtruth}
