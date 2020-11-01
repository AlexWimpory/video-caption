import csv

class GroundtruthReader:
    def __init__(self, groundtruth_filename):
        self.groundtruth_records = {}
        with open(groundtruth_filename, 'r') as fin:
            reader = csv.reader(fin)
            for row in reader:
                self.groundtruth_records[row[0]] = row[1]


    def lookup_filename(self, filename):
        return self.groundtruth_records[filename]


if __name__ == '__main__':
    #gtp = GroundtruthReader('../audio_features/data/fsd50k_dev_groundtruth.csv')
    #print(gtp.lookup_filename('407490'))
    gtp = GroundtruthReader('../audio_speech/data/librispeech_groundtruth.csv')
    print(gtp.lookup_filename("19-198-0009"))

