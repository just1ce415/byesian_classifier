"""
A Bayesian classier lib.
"""
from nltk.stem import PorterStemmer

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
        with open('stop_words.txt', 'r', encoding='utf-8') as f_ptr:
            for line in f_ptr:
                stop_words.append(line[:-1])
        bag_of_words = message.split(' ')
        for i in range(len(bag_of_words)):
            try:
                bag_of_words[i] = bag_of_words[i].lower()
                for j in range(len(bag_of_words[i])):
                    try: 
                        if bag_of_words[i][j] == '.':
                            bag_of_words[i] = bag_of_words[i].replace('.', '')
                        if bag_of_words[i][j] ==  ',':
                            bag_of_words[i] = bag_of_words[i].replace(',', '')
                        if bag_of_words[i][j] ==  '!':
                            bag_of_words[i] = bag_of_words[i].replace('!', '')
                        if bag_of_words[i][j] ==  '?':
                            bag_of_words[i] = bag_of_words[i].replace('?', '')
                        if bag_of_words[i][j] ==  ':':
                            bag_of_words[i] = bag_of_words[i].replace(':', '')
                        if bag_of_words[i][j] ==  '-':
                            bag_of_words[i] = bag_of_words[i].replace('-', '')
                        if bag_of_words[i][j] ==  '/':
                            bag_of_words[i] = bag_of_words[i].replace('/', '')
                        if bag_of_words[i][j] ==  '(':
                            bag_of_words[i] = bag_of_words[i].replace('(', '')
                        if bag_of_words[i][j] ==  ')':
                            bag_of_words[i] = bag_of_words[i].replace(')', '')
                        if bag_of_words[i][j] ==  '@':
                            bag_of_words[i] = bag_of_words[i].replace('@', '')
                        if bag_of_words[i][j] ==  '''"''':
                            bag_of_words[i] = bag_of_words[i].replace('''"''', '')
                        if bag_of_words[i][j] ==  '+':
                            bag_of_words[i] = bag_of_words[i].replace('+', '')
                        if bag_of_words[i][j] ==  '=':
                            bag_of_words[i] = bag_of_words[i].replace('=', '')
                        if bag_of_words[i][j] ==  '#':
                            bag_of_words[i] = bag_of_words[i].replace('#', '')
                        if bag_of_words[i][j] ==  '&':
                            bag_of_words[i] = bag_of_words[i].replace('&', '')
                    except IndexError:
                        break
                if bag_of_words[i] in stop_words:
                    bag_of_words.pop(i)
                    continue
                bag_of_words[i] = self.__transform_to_infinitive(bag_of_words[i])
            except IndexError:
                return bag_of_words

    def __transform_to_infinitive(self, word: str) -> str:
        """
        Returns a word casted to infinitive.
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
    print(bc.process_message('''A CBS sports writer who said Sunday that Colin Kaepernick would stand for the national anthem if he signed with another NFL team is now saying he never asked the quarterback about the issue. The confusion started after Jason La Canfora talked about Kaepernick on the network's pregame show, "The NFL Today." La Canfora said he recently spoke with the free agent about his desire to play football again. During the segment, anchor James Brown turned to the anthem debate: "And kneeling, he said?" "He's not planning on kneeling," La Canfora said. "He's going to donate all his jersey sales, and he's planning on standing for the anthem if given the opportunity." But La Canfora later said he and Kaepernick didn't talk about kneeling during the anthem, even though that answer sounded as if they did. "Standing for Anthem wasn't something that I spoke to Colin about," La Canfora wrote in a series of tweets he said were meant to "clarify" his report. "I relayed what had been reported about him standing in the future." Standing for Anthem wasn't something that I spoke to Colin about sat. I relayed what had been reported about him standing in the future... — Jason La Canfora (@JasonLaCanfora) October 8, 2017 La Canfora did not name any report in his tweets. But ESPN reported in March that Kaepernick would stand during the national anthem this season, citing unnamed sources. "Colin would have to address any future demonstrations," La Canfora tweeted. "I didn't ask him if he would sit or stand." A spokesperson for CBS sports referred CNNMoney to those tweets. La Canfora's report was picked up by several news outlets, including the Associated Press. The AP issued a correction after La Canfora tweeted about the segment. The dust-up even caught the attention of Kaepernick himself. "A lie gets halfway around the world before the truth has a chance to get its pants on," he tweeted. A lie gets halfway around the world before the truth has a chance to get its pants on. Winston S. Churchill — Colin Kaepernick (@Kaepernick7) October 8, 2017 Kaepernick also retweeted several messages that called reports that he would stand for the anthem "completely false," along with others that said he has not discussed the issue. Kaepernick played for the San Francisco 49ers until earlier this year. His free agency has sparked questions about whether it's connected to his decision to kneel during the anthem last season. Kaepernick said at the time that he was protesting the treatment of black Americans, particularly by the police.'''))