from __future__ import division
from __future__ import print_function

import csv
import sys


def redirectToFile():
    sys.stdout = open('out.txt', 'w+')


def flushToFile():
    sys.stdout.flush()


class CSVWriter:
    "log to csv file"

    def __init__(self):
        self.file = open('out.csv', 'w+')
        self.writer = csv.writer(self.file, lineterminator='\n')

    def write(self, line):
        line = [x.encode('ascii', 'replace') for x in line]  # map to ascii if possible (for py2 and windows)
        self.writer.writerow(line)
        self.file.flush()
