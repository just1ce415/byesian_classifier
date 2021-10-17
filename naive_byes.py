"""
A Bayesian classier lib.
"""
from nltk.stem import PorterStemmer
import re
import pandas as pd

class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.

    How to use:
    1) create a class prototype.
    2) process train file with process_file(), then make a train-dictinary with train() -
    the dictionary where key is every single word that is in the train file and value is
    a list [mentions in credible articles, mentions in fake articles]
    3) process test file.
    4) compute accuracy with score().
    """
    def __init__(self):
        """
        :param dictionary: dictionary with frequancy of each word(feature).
        :param num_of_trues: number of words mentioned in credible articles in total.
        """
        self.dictionary = None
        self.pstemmer = PorterStemmer()
        self.num_of_trues = 0
        self.num_of_fakes = 0

    def process_data(self, data_file):
        """
        Function for data processing and split it into X and y sets.
        :param data_file: str - train data
        :return: pd.DataFrame|list - a DF with following features: id,
        message (list, presented as bag of words), label.
        """
        df = pd.read_csv(data_file)
        del df["Unnamed: 0"]
        
        df = df[(df['Label'] == 'credible') | (df['Label'] == 'fake')]
        for index, headline, body, label in df.itertuples():
            if isinstance(df.at[index, 'Body'], str):
                df.at[index, 'Body'] = self.process_message(df.at[index, 'Body']) + self.process_message(df.at[index, 'Headline'])
            else:
                df.at[index, 'Body'] = 'NaN'
        
        df.rename(columns={'Body':'Bag of words'}, inplace=True)


        df['Label'] = df['Label'].replace(['credible'],'c')
        df['Label'] = df['Label'].replace(['fake'],'f')

        del df["Headline"]
        df = df[df['Bag of words'] != 'NaN']

        return df

    def process_message(self, message: str) -> list:
        """
        Return a processed message (bag of words as list): without punctuation,
        stopwords and all the words are stemmed.
        """
        stop_words = []
        with open('stop_words.txt', 'r', encoding='utf-8') as f_ptr:
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
            bag_of_words[i] = self.__stem_word(bag_of_words[i])
            i += 1
        return bag_of_words

    def __stem_word(self, word: str) -> str:
        """
        Returns a stemmed word.
        """
        return self.pstemmer.stem(word)

    def train(self, train_messages):
        """
        Creates train dictionary.
        :param train_messages: a DF obtained via process_data()
        :return: None
        self.dictionary = {
            word: [num_true, num_false]
        }
        """
        self.dictionary = dict()
        df = train_messages

        for sentence, label in df.itertuples(index=False):
            for word in sentence:
                if word not in self.dictionary:
                    self.dictionary[word] = [0, 0]

                if label == 'c':
                    self.dictionary[word][0] += 1
                    self.num_of_trues += 1
                elif label == 'f':
                    self.dictionary[word][1] += 1
                    self.num_of_fakes += 1

    def predict_prob(self, message):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: list - input bag of words
        :return: tuple(float, float) - probability that this message is credible and is fake
        repectively.
        NOTE: the probabily is multiblied by some 10^n to prevent data loss.
        """
        words = message
        sentence_true_prob, sentence_false_prob = 1, 1
        for word in words:

            if word in self.dictionary:
                sentence_true_prob *= ((self.dictionary[word][0] + 1) / (
                    self.num_of_trues + len(self.dictionary)) * 100000)

                sentence_false_prob *= ((self.dictionary[word][1] + 1) / (
                    self.num_of_fakes + len(self.dictionary)) * 100000)

            # If word isn't in dictionary then add it to dictionary.
            else:
                self.dictionary[word] = [1, 1]

        return (sentence_true_prob, sentence_false_prob)

    def predict(self, message):
        """
        Predict label for a given message.
        :param message: list - message (bag of words)
        :return: str - label that is most likely to be truly assigned to a given message
        (c - credible, f - fake)
        """
        true_prob, fake_prob = self.predict_prob(message)
        if true_prob >= fake_prob:
            return 'c'
        return 'f'

    def score(self, test_messages):
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param test_messages: pd.DataFrame - obtained by process_data(). 
        :return: float - percentage of accuracy.
        """
        df = test_messages

        predictions = []
        counter = 0

        bag_col = df['Bag of words']
        for value in bag_col.values:
            prediction = self.predict(value)
            predictions.append(prediction)

        df.insert(1, 'Predictions', predictions)
        pred_col = df['Predictions']

        label_col = df['Label']

        for index, value in pred_col.items():
            if value == label_col.loc[index]:
                counter += 1
            
        result = counter/len(df.index)
        return result


if __name__ == '__main__':
    bs = BayesianClassifier()
    train_df = bs.process_data("2-fake_news/train.csv")
    bs.train(train_df)

    test_df = bs.process_data('2-fake_news/test.csv')
    print(bs.score(test_df))