import os
from sys import argv
from scipy.spatial.distance import euclidean

import nltk
import numpy as np

INFINITY = 2 ** 64


def bsearch(array, pattern, lo, hi):
    if hi < lo:
        return -1
    mid = (lo + hi) / 2
    if pattern == array[mid]:
        return mid
    if pattern < array[mid]:
        return bsearch(array, pattern, lo, mid - 1)
    if pattern > array[mid]:
        return bsearch(array, pattern, mid + 1, hi)


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


def merge_counts(array):
    if len(array) == 0:
        return []

    array.sort()
    new_array = []

    pattern = array[0][0]
    counter = array[0][1]
    for entry in array[1:]:
        if entry[0] == pattern:
            counter += entry[1]
        else:
            new_array.append((pattern, counter))
            pattern = entry[0]
            counter = entry[1]
    new_array.append((pattern, counter))
    return new_array


def generate_vector_for_this_string(string, token_vectors, dimensionality):
    """
    generates the average of the vectors found for the term
    """
    counter = 0
    vector = np.zeros(dimensionality)
    for tok in nltk.wordpunct_tokenize(string.lower()):
        index = tsearch(token_vectors, tok)
        if index != -1:
            counter += 1
            vector = np.add(vector, token_vectors[index][1])
    if counter:
        return vector / counter
    return vector


def knn(vectors, token_vectors, search_string, _range, _k, dimensionality=300):
    similarity = []

    reference = generate_vector_for_this_string(search_string, token_vectors, dimensionality)

    for v in vectors:
        distance = euclidean(reference, v[1])
        if distance <= _range:
            similarity.append((distance, v[0]))
    similarity.sort()

    return similarity[:_k]


def load_string_associated_vectors(filename):
    word_vectors = []
    for line in open(filename, 'r').read().split('\n')[:-1]:
        word = line.split('\t')[0]
        vector = np.array([float(x) for x in line.split('\t')[1].split(',')])
        word_vectors.append((word, vector))
    return sorted(word_vectors)


def load_vectors(vector_file):
    if os.path.exists(vector_file):
        print('Loading vectors from', vector_file)
        for line in open(vector_file, 'r').read().split('\n')[:-1]:
            word = line.split('\t')[0]
            vec = [float(x) for x in line.split('\t')[1].split(',')]
            vectors.append((word, vec))
        return sorted(vectors)
    else:
        print('File not found:', vector_file)
        return []


""" 
do some startup related prepwork: 
require a dictionary of terms and a dictionary of vectors on startup 
build the term vectors on the fly and then run the program - keep only terms with non-zero vectors 
"""

strings = []
term_vectors = []

for i in range(1, len(argv)):
    if argv[i] == '-s' and i + 1 < len(argv):
        if os.path.exists(argv[i + 1]):
            strings = [x for x in open(argv[i + 1], 'r').read().split('\n') if len(x)]
            print(len(strings), 'strings loaded')
        else:
            print('path does not exist:', argv[i + 1])

    if argv[i] == '-v' and i + 1 < len(argv):
        if os.path.exists(argv[i + 1]):
            term_vectors = load_string_associated_vectors(argv[i + 1])
            print(len(term_vectors), 'vectors loaded')
        else:
            print('path does not exist:', argv[i + 1])

try:
    assert len(strings) and len(term_vectors)
except:
    print('string and vector inputs must be provided')
    exit()

    string_vectors = []
    for s in strings:
        v = generate_vector_for_this_string(s, term_vectors, 300)
    if sum(v):
        string_vectors.append((s, v))
print(len(string_vectors), 'string vectors generated')

_state_range = 1.0
_state_k = 20
_state_group = []
_state_label = ''
_state_previous_knn_search = ''
_state_label_stack = []
_state_recommendations = []

while True:
    string = ''
    while string == '':
        string = input('> ')

    if string.split()[0].lower() == 'range':
        if string == 'range':
            print('Current Range:', _state_range)
        elif len(string.split()) == 2 and len(string.split()[1].split('.')) <= 2 and string.split()[1].replace('.',
                                                                                                               '').isdigit():
            _state_range = float(string.split()[1])
            print('Range set to:', _state_range)
        else:
            print('To set [range], use: range [float]')

    if string.split()[0].lower() == 'k':
        if string == 'k':
            print('Current k:', _state_k)
        elif len(string.split()) == 2 and string.split()[1].isdigit():
            _state_k = int(string.split()[1])
            print('k set to:', _state_k)
        else:
            print('To set [k], use: k [integer]')

    if string.split()[0].lower() == 'knn':
        if 1 < len(string.split()):
            search_string = ' '.join(string.split()[1:])
            _state_group = knn(string_vectors, term_vectors, search_string, _state_range, _state_k)
            _state_previous_knn_search = search_string
            print(sorted([str(x[1]) for x in _state_group]))
        elif len(_state_previous_knn_search):
            print('Searching previous string: knn', _state_previous_knn_search)
            _state_group = knn(string_vectors, term_vectors, search_string, _state_range, _state_k)
            print(sorted([str(x[1]) for x in _state_group]))
        else:
            print('type help to get information')

    if string.lower() == 'group':
        print('Currently selected items:')
        print(sorted([str(x[1]) for x in _state_group]))

    if string.lower() == 'detail':
        print('Currently selected items:')
        print(_state_group)

    if string.split()[0].lower() == 'label':
        if string.lower() == 'label':
            print('Current Label:', _state_label)
        elif 1 < len(string.split()):
            _state_label = ' '.join(string.split()[1:])
            print('Label set to:', _state_label)

    if string.split()[0].lower() == 'assign':
        if len(_state_group) and len(_state_label):
            _state_label_stack.append((_state_label, [x[1] for x in _state_group]))
            print('Assigning Label(', _state_label, ') to', len(_state_group), 'items')
        else:
            print('Label must be specified and group must be populated before assignment can occur')

    if string.lower() == 'stack':
        print('Label Stack:')
        print('\n'.join([str((x[0], len(x[1]))) for x in _state_label_stack]))

    if string.split()[0].lower() == 'export':
        if 1 < len(string.split()):
            export_filename = ' '.join(string.split()[1:])
            print('Exporting Labels To:', export_filename)

            save = open(export_filename, 'w')
            for group in _state_label_stack:
                label = group[0]
                for item in group[1]:
                    save.write(str(item) + '\t' + str(label) + '\n')
            save.close()

    if string.lower() == 'emit':
        if len(_state_group):
            members = sorted([x[1] for x in _state_group])
            item_stack = []
            for item in [x[1] for x in _state_group]:
                group = [x for x in knn(string_vectors, term_vectors, item, _state_range, INFINITY) if
                         bsearch(members, x[1], 0, len(members) - 1) == -1]
                item_stack = merge_counts(item_stack + [(x[1], 1) for x in group])
                print('Searching:', item)
                print(list(reversed(sorted([(x[1], x[0]) for x in item_stack if 1 < x[1]])))[:100])
            _state_recommendations = list(reversed(sorted([(x[1], x[0]) for x in item_stack if 1 < x[1]])))[:100]

        else:
            print('Must run knn first')

    if string.split()[0].lower() == 'preview':
        if string == 'preview':
            print('Current Recommendations:')
            print('\n'.join([str(x) for x in _state_recommendations]))
        elif len(string.split()) == 2 and string.split()[1].isdigit():
            items_to_show = int(string.split()[1])
            print('\n'.join([str(x) for x in _state_recommendations][:items_to_show]))

    if string.split()[0].lower() == 'keep':
        if string == 'keep':
            print('Current Recommendations:')
            print('\n'.join([str(x) for x in _state_recommendations]))
        elif len(string.split()) == 2 and string.split()[1].isdigit():
            items_to_keep = int(string.split()[1])
            _state_group += [x for x in _state_recommendations if items_to_keep <= x[0]]
            print(sorted([str(x[1]) for x in _state_group]))

    if string.split()[0].lower() == 'add':
        if 1 < len(string.split()):
            search_string = ' '.join(string.split()[1:])

            index_a = tsearch(string_vectors, _state_previous_knn_search)
            index_b = tsearch(string_vectors, search_string)
            if index_a != -1 and index_b != -1:
                print('Adding', search_string, 'to the group')
                distance = euclidean(string_vectors[index_a][1], string_vectors[index_b][1])
                _state_group = sorted(_state_group + [(distance, string_vectors[index_b][0])])
                print(sorted([str(x[1]) for x in _state_group]))
            else:
                print('Add failed because one of the words was missing')

    if string.split()[0].lower() == 'remove':
        if 1 < len(string.split()):
            search_string = ' '.join(string.split()[1:])
            print('Removing', search_string, 'from group')
            _state_group = [x for x in _state_group if x[1] != search_string]
            print(sorted([str(x[1]) for x in _state_group]))

    if string.split()[0].lower() == 'pop':
        if 1 < len(string.split()):
            search_string = ' '.join(string.split()[1:])
            print('Removing', search_string, 'from stack')
            _state_label_stack = [x for x in _state_label_stack if x[0] != search_string]
            print('\n'.join([str((x[0], len(x[1]))) for x in _state_label_stack]))

            # TODO write a thing in the help menu for this
    if string.split()[0].lower() == 'visualize':

        # print _state_label_stack

        labels = []
        for stack in _state_label_stack:
            labels += [(x, stack[0]) for x in stack[1]]
        labels.sort()

        vectors = []
        for l in labels:
            vectors.append(generate_vector_for_this_string(l[1], token_vectors, 300))

        from sklearn.decomposition import PCA

        pca_result = PCA(n_components=20).fit_transform(vectors)

        from sklearn import manifold

        viz = manifold.TSNE().fit_transform(pca_result)

        print(labels)

        save = open('exported vis labels', 'w')
        for i in range(len(viz)):
            save.write(str(viz[i][0]) + '\t' + str(viz[i][1]) + '\t' + labels[i][0] + '\n')
        save.close()

    if string.lower() == 'help':
        print('range            - display the current search range')
        print('range [float]    - set the search range to [float]')
        print('k                - display the current number of neighbors')
        print('k [int]          - set the number of neighbors to [int]')
        print('knn              - run k-nearest neighbors with previous search string')
        print('knn [string]     - run k-nearest neighbors on [string]')
        print('group            - display the working group')
        print('detail           - display the working group in more detail')
        print('label            - display the current label')
        print('label [string]   - set the current label to [string]')
        print('assign           - assign the current label to the working group')
        print('stack            - display the current label stack')
        print('export [string]  - export the label stack to a file')
        print('emit             - try and spread labels to neighbors')
        print('preview          - show emit recommendations')
        print('preview [int]    - display the first [int] recommendations')
        print('keep             - same as preview')
        print('keep [int]       - commit items with [int] occurrences to working group')
        print('add [string]     - add [string] to the working group')
        print('remove [string]  - remove item [string] from working group')
        print('pop [string]     - remove group [string] from stack')