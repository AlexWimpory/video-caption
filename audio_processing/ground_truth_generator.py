import csv
import os


def process_tut_groundtruth():
    """Processing TUT audio_processing record to standard format"""
    with open('../audio_features/data/tut_meta.txt', 'r') as fin, open('../audio_features/data/tut_groundtruth.csv', 'w', newline='') as fout:
        reader = csv.reader(fin, delimiter='\t')
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow([row[0].strip().replace('audio/', ''), [row[1].strip()]])


def process_fsd50k_groundtruth():
    """Processing FSD50K audio_processing record to standard format"""
    with open('../audio_features/data/fsd50k_dev.csv', 'r') as fin, open(
            '../audio_features/data/fsd50k_dev_groundtruth.csv', 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            labels = row[1].replace('"[\'', '').split(',')
            writer.writerow([row[0].strip(), labels])


def process_librispeech_groundtruth():
    """Processing librispeech audio_processing record to standard format"""
    with open('../audio_speech/data/librispeech_groundtruth.csv', 'w', newline='') as fout:
        path = 'D:\\Audio Speech\\LibriSpeech\\train-clean-100'
        writer = csv.writer(fout)

        for root, dirs, files in os.walk(path):
            for file in files:
                if(file.endswith(".txt")):
                    filelist = os.path.join(root, file)
                    with open(filelist, 'r') as fin:
                        for row in fin:
                            (address, data) = row.split(sep=' ', maxsplit=1)
                            writer.writerow([address, data.strip()])


if __name__ == '__main__':
    process_tut_groundtruth()
    process_fsd50k_groundtruth()
    process_librispeech_groundtruth()