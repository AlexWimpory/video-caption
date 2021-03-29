import csv
import ast

"""Look up the labels for the file in the ground truth document"""


class GroundtruthReader:
    def __init__(self, groundtruth_filename):
        self.groundtruth_records = {}
        with open(groundtruth_filename, 'r') as fin:
            reader = csv.reader(fin)
            for row in reader:
                if row[1].startswith('[') and row[1].endswith(']'):
                    self.groundtruth_records[row[0]] = ast.literal_eval(row[1])
                else:
                    self.groundtruth_records[row[0]] = row[1]

    def lookup_filename(self, filename):
        return self.groundtruth_records[filename]


if __name__ == '__main__':
    gtp = GroundtruthReader('../audio_sounds/data/fsd50k_dev_groundtruth.csv')
    print(gtp.lookup_filename('407490'))
    #gtp = GroundtruthReader('../audio_speech/data/librispeech_groundtruth.csv')
    #print(gtp.lookup_filename("19-198-0009"))

