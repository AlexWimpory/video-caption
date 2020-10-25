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


# def find_labels(groundtruth_filename, filename):
#     with open(groundtruth_filename, 'r') as fin:
#         reader = csv.reader(fin)
#         for row in reader:
#             if row[0] == filename:
#                 return row[1]
#         return None


if __name__ == '__main__':
    gtp = GroundtruthReader('data/fsd50k_dev_groundtruth.csv')
    print(gtp.lookup_filename('407490'))
    #print(find_labels('data/tut_groundtruth.csv', 'b020_0_10.wav'))