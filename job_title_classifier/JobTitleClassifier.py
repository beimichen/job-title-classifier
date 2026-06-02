import random
import time
import nltk
import numpy as np

from job_title_classifier.euclid_util import bsearch, tsearch
from job_title_classifier.euclid_util import load_string_associated_vectors
from scipy.spatial.distance import euclidean
import logging
# import boto3
# from botocore.exceptions import ClientError
import pickle
import io


class JobTitleClassifier():
    """
    This class builds vector representations of input strings and then provides on-demand classification of strings supplied through the API
    This is implemented as follows:

    Setup:
        1)  A set of reference strings is supplied -- these are representative of the strings the developer will try to classify
        2)  A set of tokens with associated vectors is supplied -- tokenization as performed by the wordpunct tokenizer from nltk
        3)  The reference strings are tokenized and vectorized at initialization -- this allows for vector dictionary substitution

    Usage:
        1)  A classification request is made and thresholds for search distance, minimum neighbors, and minimum confidence are specified
        2)  The sorted reference strings are first searched for an exact pattern match -- this precludes multiple class membership for reference strings (potential bug)
        3)  If an exact match is not found, a knn classifier is used with the supplied thresholds
            If a match is found in this way, the result and the predicted class membership are written to a file for manual confirmation
            Class membership predictions that are manually verified are added to the list of reference strings

    Additional Functionality:
        *   A batch process will be specified somewhere -- this will allow the user to automate processing on a batch of input strings
        *   Testing functionality will be provided via a JobTitleClassifierTest class and will provide the following functionality:
            a)  Threshold optimization -- the labeled reference set will be subdivided into training and test sets to search for settings that yield the highest accuracies
    """

    def __init__(self, reference_strings=[], token_vectors=[], dimensionality=300):
        """
        @param:
            reference_strings   (string, classification)
            token_vectors       (token, vector)
            reference_vectors   ((string, classification), composite_vector)
            dimensionality      int

        """

        # copy some parameters
        self.reference_strings = reference_strings
        self.reference_classes = sorted(list(set([x[1] for x in self.reference_strings])))
        self.token_vectors = token_vectors
        self.reference_vectors = []
        self.dimensionality = dimensionality

        # do some setup
        self.prepare_reference_vectors()
        self.output_filename = 'JobTitleClassifier_predicted_classes_' + str(int(time.time()))
        # set up some really basic feedback/logging
        # self.debug = open(self.output_filename, 'w')
        # input("ok")

    def classify(self, search_string, _range=5.0, _k=20, confidence=0.90):
        stream = io.BytesIO()

        class_likelihoods = np.zeros(len(self.reference_classes))

        # stage 1 - check to see if we have an exact pattern match - it's short enough that i won't break it up
        string_index = tsearch(self.reference_strings, search_string)
        if string_index != -1:
            class_index = bsearch(self.reference_classes, self.reference_strings[string_index][1])
            assert class_index != -1
            class_likelihoods[class_index] = 1
            return class_likelihoods

            # stage 2 - use k-nearest neighbors to identify the most likely class membership
        similarity = []

        search_string_vector = self.generate_vector_for_this_string(search_string)

        if sum(search_string_vector) == 0:
            return list(zip(np.zeros(len(self.reference_classes)), self.reference_classes))

        for refvec in self.reference_vectors:
            distance = euclidean(search_string_vector, refvec[1])
            if distance <= _range:
                similarity.append([distance] + refvec[0])
        similarity.sort()

        for sim in similarity[:_k]:
            index = bsearch(self.reference_classes, sim[-1])
            assert index != -1
            class_likelihoods[index] += 1

        if sum(class_likelihoods) and _k <= len(similarity):
            class_likelihoods /= sum(class_likelihoods)
            if confidence <= max(class_likelihoods):

                log_string = search_string.replace('\t', '[TAB]').replace('\n', '[RET]') + '\t' + list(reversed(sorted(zip(class_likelihoods, self.reference_classes))))[0][1] + '\n'
                # print("LOG STRING")
                # print(log_string)
                pickled_log = pickle.dumps(log_string)
                # print(pickled_log)
                # randnum = random.randint(1, 1000000)
                # final_filename = self.output_filename + "_" + str(randnum) + ".txt"


                # self.debug.write(search_string.replace('\t', '[TAB]').replace('\n', '[RET]') + '\t' +
                #                  list(reversed(sorted(zip(class_likelihoods, self.reference_classes))))[0][1] + '\n')

                return list(zip(class_likelihoods, self.reference_classes))


        # upload to s3

        # pickle.dumps(obj)

        # with open(self.output_filename, "rb") as f:
        #     s3.upload_fileobj(f, "BUCKET_NAME", "OBJECT_NAME")

        return list(zip(np.zeros(len(self.reference_classes)), self.reference_classes))

    def prepare_reference_vectors(self):
        if len(self.reference_strings) == 0 or len(self.token_vectors) == 0:
            print('Unable to initialize reference vectors because reference strings or token vectors are empty')
            return []

        self.reference_vectors = []
        for ref in self.reference_strings:
            vector = self.generate_vector_for_this_string(ref[0])
            if sum(vector):
                self.reference_vectors.append((ref, vector))
        # print(len(self.reference_vectors), 'reference vectors generated')

    def generate_vector_for_this_string(self, string):
        """
        generates the average of the vectors found for the term
        """
        counter = 0
        vector = np.zeros(self.dimensionality)
        for tok in nltk.wordpunct_tokenize(string.lower()):
            index = tsearch(self.token_vectors, tok)
            if index != -1:
                counter += 1
                vector = np.add(vector, self.token_vectors[index][1])
        if counter:
            return vector / counter
        return vector


""" 
build a local model for some preliminary testing 
"""

''' 
strings = sorted([x.split('\t') for x in open('recon - classifier labels', 'r').read().split('\n') if len(x)]) 
vectors = load_string_associated_vectors('recon - stage one vectors') 


jt = JobTitleClassifier(strings, vectors) 

for title in open('recon - master job titles','r').read().split('\n')[:-1]: 
    classification = jt.classify(title) 
    print classification 

'''