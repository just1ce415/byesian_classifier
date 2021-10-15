"""
A Bayesian classier lib.
"""
from nltk.stem import PorterStemmer
import re

class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.

    How to use:
    1) create a class prototype.
    2) process train file with process_file() and make a dictinary with train() -
    fits messages to respective labels.
    3) process test file.
    4) compute accuracy with score.
    """
    def __init__(self):
        """
        :param dictionary: dictionary with frequancy of each word(feature).
        """
        self.dictionary = None
        self.pstemmer = PorterStemmer()

    def process_data(self, data_file):
        """
        Function for data processing and split it into X and y sets.
        :param data_file: str - train data
        :return: pd.DataFrame|list - a DF/list with following feautures: id,
        message (presented as bag of words), label.
        """
        pass

    def process_message(self, message: str) -> list:
        """
        Return a processed message (bag of words as list): without punctuation,
        stopwords and all the words casted to infinitives
        """
        stop_words = []
        with open('byesian_classifier/stop_words.txt', 'r', encoding='utf-8') as f_ptr:
            for line in f_ptr:
                stop_words.append(line[:-1])
        stop_words.append("")
        bag_of_words = message.split(' ')
        i = 0
        while (i < len(bag_of_words)):
            bag_of_words[i] = bag_of_words[i].lower()
            # Exclude punctuation
            match = re.search(r"[a-zA-Z']+", bag_of_words[i])
            if match:
                bag_of_words[i] = match.group(0)
            else:
                bag_of_words.pop(i)
                continue
            # Check for unrelevent apostrosphe
            if bag_of_words[i][0] == "'":
                bag_of_words[i] = bag_of_words[i][1:]
            elif bag_of_words[i][-1] == "'":
                bag_of_words[i] = bag_of_words[i][:-1]
            # Exclude stop-words
            if bag_of_words[i] in stop_words:
                bag_of_words.pop(i)
                continue
            # Stemming
            bag_of_words[i] = self.__transform_to_infinitive(bag_of_words[i])
            i += 1
        return bag_of_words

    def __transform_to_infinitive(self, word: str) -> str:
        """
        Returns a word casted to infinitive (stemmed).
        """
        return self.pstemmer.stem(word)

    def train(self, train_messages):
        """
        Creates train dictionary.
        :param train_messages: a DF/list obtained via process_data
        :return: None
        """
        pass

    def predict_prob(self, message, label):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        pass

    def predict(self, message):
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        pass

    def score(self, test_messages):
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param test_messages: pd.DataFrame|list - obtained by process_data(). 
        :return: float - percentage of accuracy.
        """
        pass


if __name__ == '__main__':
    bc = BayesianClassifier()
    print(bc.process_message("bbc'"))