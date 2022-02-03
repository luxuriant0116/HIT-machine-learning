import os
import struct
import numpy as np
import csv


def load_mnist():
    """Load MNIST data from `path`"""
    labels_path = './t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'
    images_path = './t10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def load_fer():
    path = 'test.csv'
    with open(path) as f:
        reader = csv.reader(f)
        row_id = 0
        faces = []
        for row in reader:
            if row_id == 0:
                row_id += 1
                continue
            else:
                face = [float(s) for s in row[0].split(' ')]
                faces.append(face)
    return np.array(faces)


if __name__ == '__main__':
    load_fer()
