import numpy as np


def bsearch(array, pattern):
    return _bsearch(array, pattern, 0, len(array) - 1)


def _bsearch(array, pattern, lo, hi):
    if hi < lo:
        return -1
    mid = (lo + hi) // 2
    if pattern == array[mid]:
        return mid
    if pattern < array[mid]:
        return _bsearch(array, pattern, lo, mid - 1)
    if pattern > array[mid]:
        return _bsearch(array, pattern, mid + 1, hi)


def tsearch(array, pattern):
    return _tsearch(array, pattern, 0, len(array) - 1)


def _tsearch(array, pattern, lo, hi):
    if hi < lo:
        return -1
    mid = (lo + hi) // 2
    if pattern == array[mid][0]:
        return mid
    if pattern < array[mid][0]:
        return _tsearch(array, pattern, lo, mid - 1)
    if pattern > array[mid][0]:
        return _tsearch(array, pattern, mid + 1, hi)


def load_string_associated_vectors(filename):
    word_vectors = []
    for line in open(filename, 'r').read().split('\n')[:-1]:
        word = line.split('\t')[0]
        vector = np.array([float(x) for x in line.split('\t')[1].split(',')])
        word_vectors.append((word, vector))
    return sorted(word_vectors) 