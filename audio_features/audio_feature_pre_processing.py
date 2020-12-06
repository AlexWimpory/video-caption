from pandas import DataFrame
from audio_processing.ground_truth_processor import GroundtruthReader
from mfcc_creator import create_mfcc, mfcc_mean
from file_utils import return_from_path, save_object
from functools import partial
import os


def prepare_audio_feature(groundtruth, file_name):
    """Generate an MFCC and look up the labels which are matched together"""
    base_file_name = os.path.splitext(os.path.basename(file_name))[0]
    gtp = GroundtruthReader(groundtruth)
    from_groundtruth = gtp.lookup_filename(base_file_name)
    mfcc = mfcc_mean(create_mfcc(file_name))
    return {'mfcc': mfcc, 'labels': from_groundtruth}


if __name__ == '__main__':
    prepare_audio_feature_groundtruth = partial(prepare_audio_feature, 'data/UrbanSound8K_groundtruth.csv')
    ftrs = return_from_path(prepare_audio_feature_groundtruth,
                            'D:\\Audio Features\\Small',
                            '.wav')
    audio_feature_df = DataFrame(ftrs)
    save_object(audio_feature_df, 'data/UrbanSound8K_all_dataframe.data')

