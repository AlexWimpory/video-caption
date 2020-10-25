import csv

def process_tut_groundtruth():
    """Processing TUT groundtruth record to standard format"""
    with open('data/tut_meta.txt', 'r') as fin, open('data/tut_groundtruth.csv', 'w', newline='') as fout:
        reader = csv.reader(fin, delimiter='\t')
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow([row[0].strip().replace('audio/', ''), [row[1].strip()]])


def process_fsd50k_groundtruth():
    """Processing FSD50K groundtruth record to standard format"""
    with open('data/fsd50k_dev.csv', 'r') as fin, open('data/fsd50k_dev_groundtruth.csv', 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            labels = row[1].replace('"[\'', '').split(',')
            writer.writerow([row[0].strip(), labels])


if __name__ == '__main__':
    process_tut_groundtruth()
    process_fsd50k_groundtruth()