import csv
import os
import re
from os.path import splitext, basename
from num2words import num2words
from file_utils import append_to_file_name


def process_tut_ground_truth():
    """Process TUT record to standard format"""
    with open('../audio_features/data/tut_meta.txt', 'r') as fin, open('../audio_features/data/tut_groundtruth.csv', 'w', newline='') as fout:
        reader = csv.reader(fin, delimiter='\t')
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow([row[0].strip().replace('audio/', '').replace('.wav', ''), [row[1].strip()]])


def process_fsd50k_ground_truth():
    """Process FSD50K record to standard format"""
    with open('../audio_features/data/fsd50k_dev.csv', 'r') as fin, open(
            '../audio_features/data/fsd50k_dev_groundtruth.csv', 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            labels = row[1].replace('"[\'', '').split(',')
            writer.writerow([row[0].strip(), labels])


def process_librispeech_ground_truth():
    """Process librispeech record to standard format"""
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


def process_urbansound8k_ground_truth():
    """Process urbansound8k record to standard format"""
    with open('../audio_features/data/UrbanSound8K.csv', 'r') as fin, open(
            '../audio_features/data/UrbanSound8K_groundtruth.csv', 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow([row[0].strip().replace('.wav', ''), [row[7].strip()]])


def process_ted_ground_truth(path, audio_file_path):
    """Process TED talk transcript to standard format"""
    new_file_name = append_to_file_name(path, '_groundtruth', '.csv')
    with open(path, 'r') as fin, open( new_file_name, 'w') as fout:
        names = []
        fout.write(splitext(basename(audio_file_path))[0] + ',')
        while True:
            line = fin.readline()
            if not line:
                break
            grammer_removed = line.replace('"', '').replace('!', '').replace('?', '')\
                .replace('.', '').replace('-', ' ').replace(',', '')
            brackets_removed = re.sub('\(.*?\)', '', grammer_removed)
            times_removed = re.sub('\d\d:\d\d\n', '', brackets_removed)
            number_list = re.findall('[0-9]+', times_removed)
            numbers_removed = times_removed
            for number in number_list:
                numbers_removed = numbers_removed.replace(number, num2words(int(number)))
            numbers_removed = numbers_removed.replace('-', ' ').replace(',', '').replace(';', '').upper().strip()
            pos = numbers_removed.find(':')
            names_removed = numbers_removed
            if pos != -1 and pos < 20:
                names.append(numbers_removed[0:pos+1])
                names_removed = names_removed.replace(numbers_removed[0:pos+1], '')
            names_removed = ' '.join(names_removed.replace(':', '').split())
            if len(names_removed) > 0:
                fout.write(names_removed + ' ')
        print(f'Removed authors: {names}')


if __name__ == '__main__':
    #process_tut_ground_truth()
    #process_fsd50k_ground_truth()
    #process_librispeech_ground_truth()
    #process_urbansound8k_ground_truth()
    process_ted_ground_truth('D:\\TED\\CKWilliams_2001-480p-Transcipt.txt', 'D:\\TED\\CKWilliams_2001-480p.wav')