import os
import re
import xml.etree.ElementTree as ET
from sys import argv
from string import punctuation
from abc import ABC, abstractmethod
from math import log
from more_itertools import pairwise, triplewise
from random import choices
from time import process_time


class Token:

    def __init__(self, tag, c5, hw, pos, text):
        self.pos = pos
        self.c5 = c5
        self.hw = hw
        self.text = text
        self.tag = tag


class Sentence:

    def __init__(self, index=0):
        self.index = index
        self.tokens = []
        self.length = 0
        self.num_of_tokens = 0

    def __str__(self):
        result = ""
        for token in self.tokens:
            if token.text is None:
                continue
            result += token.text
            result += " "
        return result

    def add_token(self, token: Token):
        self.tokens.append(token)
        self.num_of_tokens += 1

    def insert_token(self, token: Token, index: int):
        self.tokens.insert(0, token)
        self.num_of_tokens += 1


class Corpus:

    def __init__(self):
        self.documents = []
        self.sentences = []
        self.num_of_tokens = 0
        self.sentence_index = 1
        self.distinct_words = dict()
        self.bigram_dict = dict()
        self.trigram_dict = dict()
        self.lengths_dict = dict()
        self.lengths_probabilities = dict()
        self.unigram_probabilities_dict = dict()
        self.bigram_options_dict = dict()
        self.trigram_options_dict = dict()

    def __str__(self):
        result = ""
        for sentence in self.sentences:
            sentence_str = str(sentence)
            if sentence_str.startswith('='):
                result += "\n"
                result += str(sentence_str)
                result += "\n"
                result += "\n"
            else:
                result += str(sentence_str)
                result += "\n"
        return result

    def get_vocabulary_size(self) -> int:
        return len(self.distinct_words.keys())

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)
        self.num_of_tokens += sentence.num_of_tokens
        if self.lengths_dict.get(sentence.length) is None:
            self.lengths_dict[sentence.length] = 0
        self.lengths_dict[sentence.length] += 1

    def add_document(self, document: str):
        self.documents.append(document)

    def add_word_to_distinct_dict(self, word):
        if self.distinct_words.get(word) is None:
            self.distinct_words[word] = 1
        else:
            self.distinct_words[word] += 1

    def tokenize_xml_element(self, element: ET.Element) -> Token:
        tag = element.tag
        c5 = element.attrib.get('c5')
        hw = element.attrib.get('hw')
        pos = element.attrib.get('pos')
        text = element.text
        if text is not None:
            text = text.strip()
        return Token(tag, c5, hw, pos, text)

    def tokenize_from_text(self, word: str) -> Token:
        tag = 'c' if word in punctuation else 'w'
        c5 = 'PUN' if word in punctuation else None
        return Token(tag, c5, None, None, word)

    def add_sentence_from_xml_element(self, element: ET.Element) -> None:
        sentence = Sentence(self.sentence_index)
        for child in element:
            if len(child) > 0:
                for word in child:
                    token = self.tokenize_xml_element(word)
                    sentence.add_token(token)
                    if token.text is not None:
                        sentence.length += len(token.text)
                continue
            token = self.tokenize_xml_element(child)
            sentence.add_token(token)
            if token.text is not None:
                sentence.length += len(token.text)
                self.add_word_to_distinct_dict(token.text.lower())
        self.add_sentence(sentence)
        self.add_tokens_to_bigram(sentence.tokens)
        self.add_tokens_to_trigram(sentence.tokens)
        self.sentence_index += 1

    def split_text_to_sentences(self, text: str) -> []:
        result = []
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=[.?])\s', text)
        for sentence in sentences:
            if '\n' in sentence:
                lines = sentence.splitlines()
                result.extend([line for line in lines if line != ""])
            else:
                result.append(sentence)
        return result

    def split_sentence_to_tokens(self, sentence: str) -> []:
        initial_split = re.split('(\W)', sentence)
        return [token for token in initial_split if token.strip()]

    def add_xml_file_to_corpus(self, file_name: str):
        tree = ET.parse(file_name)
        root = tree.getroot()
        for sentence in root.iter('s'):
            self.add_sentence_from_xml_element(sentence)
        self.add_document(self.get_xml_document_title(root))

    def get_xml_document_title(self, root):
        header_tag = root.find('teiHeader')
        description_tag = header_tag.find('fileDesc')
        title_stmt_tag = description_tag.find('titleStmt')
        title_tag = title_stmt_tag.find('title')
        return title_tag.text

    def add_text_file_to_corpus(self, file_name: str):
        with open(file_name, encoding="utf8") as text_file:
            self.add_document(self.get_text_document_title(file_name))
            text = text_file.read()
            sentences = self.split_text_to_sentences(text)
            for sentence in sentences:
                tokens = self.split_sentence_to_tokens(sentence)
                sentence_to_add = Sentence(self.sentence_index)
                for token in tokens:
                    tokenized = self.tokenize_from_text(token)
                    sentence_to_add.add_token(tokenized)
                    if tokenized.text is not None and tokenized.tag == 'w':
                        self.add_word_to_distinct_dict(tokenized.text)
                self.add_sentence(sentence_to_add)
                self.sentence_index += 1

    def create_text_file(self, file_name: str):
        with open(file_name, encoding="utf8", mode="w") as file:
            file.write(str(self))
        print(f"Successfully written to {file_name}")

    def get_text_document_title(self, file_name):
        name = file_name.split(os.sep)[-1]
        name_without_ext = name.split('.')[0]
        return f"{name_without_ext} - Wiki page"

    def add_tokens_to_bigram(self, tokens: list) -> None:
        bi_list = list(pairwise(tokens))
        for pair in bi_list:
            if pair[0].text is None or pair[1].text is None:
                continue
            first_token = pair[0].text.lower()
            second_token = pair[1].text.lower()
            key = (first_token, second_token)
            if self.bigram_dict.get(key) is None:
                self.bigram_dict[key] = 0
            self.bigram_dict[key] += 1
            self.add_tokens_to_bigram_options_dict(first_token, second_token)

    def add_tokens_to_bigram_options_dict(self, first_token: str, second_token: str) -> None:
        self.add_tokens_to_options_dict(self.bigram_options_dict, first_token, second_token)

    def add_tokens_to_trigram(self, tokens: list) -> None:
        tri_list = list(triplewise(tokens))
        for triplet in tri_list:
            token_1 = triplet[0].text
            token_2 = triplet[1].text
            token_3 = triplet[2].text
            if token_1 is None or token_2 is None or token_3 is None:
                continue
            key = (token_1.lower(), token_2.lower(), token_3.lower())
            if self.trigram_dict.get(key) is None:
                self.trigram_dict[key] = 0
            self.trigram_dict[key] += 1
            self.add_tokens_to_trigram_options_dict(token_1.lower(), token_2.lower(), token_3.lower())

    def add_tokens_to_trigram_options_dict(self, token_1: str, token_2: str, token_3: str) -> None:
        key = (token_1, token_2)
        self.add_tokens_to_options_dict(self.trigram_options_dict, key, token_3)

    def add_tokens_to_options_dict(self, options_dict: dict, key, token: str) -> None:
        if options_dict.get(key) is None:
            options_dict[key] = {token: 1}
        else:
            tokens_dict = options_dict[key]
            if tokens_dict.get(token) is None:
                tokens_dict[token] = 0
            tokens_dict[token] += 1

    # This method adds the <B> & <E> tags, and updates all required dictionaries as well.
    def add_begin_and_end_tokens(self) -> None:
        begin = Token("Beginning", None, None, None, "<B>")
        end = Token("End", None, None, None, "<E>")
        for sentence in self.sentences:
            sentence.insert_token(begin, 0)
            sentence.add_token(end)
            self.add_word_to_distinct_dict(begin.text)
            self.add_word_to_distinct_dict(end.text)
            self.add_tokens_to_bigram([sentence.tokens[0], sentence.tokens[1]])
            self.add_tokens_to_bigram([sentence.tokens[-2], sentence.tokens[-1]])
            self.add_tokens_to_trigram([sentence.tokens[0], sentence.tokens[1], sentence.tokens[2]])
            self.add_tokens_to_trigram([sentence.tokens[-3], sentence.tokens[-2], sentence.tokens[-1]])
        self.init_probability_dicts()

    def update_length_probabilities(self) -> None:
        sentences_count = len(self.sentences)
        for key in self.lengths_dict.keys():
            key_length = self.lengths_dict[key]
            self.lengths_probabilities[key] = key_length / sentences_count

    def update_unigram_probabilities(self) -> None:
        for key in self.distinct_words.keys():
            occurrences = self.distinct_words[key]
            self.unigram_probabilities_dict[key] = occurrences / self.num_of_tokens

    def update_bigram_probabilities(self) -> None:
        for key in self.bigram_options_dict.keys():
            options_dict = self.bigram_options_dict[key]
            occurrences = sum(options_dict.values())
            for token in options_dict.keys():
                options_dict[token] /= occurrences

    def init_probability_dicts(self) -> None:
        self.update_unigram_probabilities()
        self.update_bigram_probabilities()
        self.update_trigram_probabilities()

    def update_trigram_probabilities(self) -> None:
        for key in self.trigram_options_dict.keys():
            options_dict = self.trigram_options_dict[key]
            occurrences = sum(options_dict.values())
            for token in options_dict.keys():
                options_dict[token] /= occurrences


# Implement an n-gram language model class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class):

class NGramModelBase(ABC):

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.uni_dict = self.corpus.distinct_words
        self.bi_dict = self.corpus.bigram_dict
        self.tri_dict = self.corpus.trigram_dict
        self.voc_size = corpus.get_vocabulary_size()
        self.total_tokens = corpus.num_of_tokens
        self.num_of_random_sentences = 5

    @abstractmethod
    def get_sentence_probability(self, sentence: str):
        pass

    @abstractmethod
    def get_model_title(self) -> str:
        pass

    @abstractmethod
    def generate_random_sentence(self, length: int) -> str:
        pass

    def unigram_smooth(self, token: str) -> float:
        numerator = self.uni_dict.get(token.lower(), 0) + 1
        denominator = self.total_tokens + len(self.uni_dict.keys())
        return log(numerator / denominator)

    def bigram_smooth(self, token_1: str, token_2: str) -> float:
        key = (token_1.lower(), token_2.lower())
        numerator = self.bi_dict.get(key, 0) + 1
        token_2_count = self.uni_dict.get(token_2.lower(), 0)
        denominator = token_2_count + len(self.bi_dict.keys())
        return log(numerator / denominator)

    def trigram_smooth(self, token_1: str, token_2: str, token_3: str) -> float:
        key = (token_1.lower(), token_2.lower(), token_3.lower())
        numerator = self.tri_dict.get(key, 0) + 1
        bigram_count = self.bi_dict.get((token_2, token_3), 0)
        denominator = bigram_count + len(self.tri_dict.keys())
        return log(numerator / denominator)

    def tokenize_sentence(self, sentence):
        tokens = self.corpus.split_sentence_to_tokens(sentence)
        for i in range(len(tokens)):
            if tokens[i] == '’' and i != len(tokens) - 1 and i - 1 >= 0:
                if tokens[i - 1].endswith('n') and tokens[i + 1] == 't':
                    tokens[i - 1] = tokens[i - 1][:-1]
                    tokens[i + 1] = f'n\'{tokens[i + 1]}'
                else:
                    tokens[i + 1] = f'\'{tokens[i + 1]}'
        tokens = [tok for tok in tokens if tok != '’']
        return tokens

    def generate_random_bigram_token(self, prev_token: str) -> str:
        options = list(self.corpus.bigram_options_dict[prev_token.lower()].keys())
        probabilities = list(self.corpus.bigram_options_dict[prev_token.lower()].values())
        next_token = choices(options, weights=probabilities, k=1)[0]
        return next_token

    def generate_random_trigram_token(self, prev_tokens: tuple) -> str:
        options = list(self.corpus.trigram_options_dict[prev_tokens].keys())
        probabilities = list(self.corpus.trigram_options_dict[prev_tokens].values())
        next_token = choices(options, weights=probabilities, k=1)[0]
        return next_token

    def get_sentences_probability(self, sentences: list) -> None:
        result = ""
        result += f"{self.get_model_title()}:\n\n"
        for s in sentences:
            result += s
            result += '\n'
            result += str(self.get_sentence_probability(s))
            result += '\n'
            result += '\n'
        return result

    def get_random_sentences(self, length: int) -> None:
        result = ""
        result += f"{self.get_model_title()}:\n"
        for i in range(self.num_of_random_sentences):
            result += self.generate_random_sentence(length)
            result += '\n'
        return result


class UnigramModel(NGramModelBase):

    def __init__(self, corpus: Corpus):
        super().__init__(corpus)

    def get_sentence_probability(self, sentence: str):
        tokens = self.tokenize_sentence(sentence)
        probability = 0
        for token in tokens:
            probability += self.unigram_smooth(token)
        return probability

    def get_model_title(self) -> str:
        return "Unigrams Model"

    def generate_random_sentence(self, length: int) -> str:
        corpus.update_unigram_probabilities()
        tokens = ["<b>"]
        ccs = choices(list(self.corpus.unigram_probabilities_dict.keys()),
                      weights=list(self.corpus.unigram_probabilities_dict.values()), k=length - 1)
        for tok in ccs:
            tokens.append(tok)
            if tok == "<e>" or tok == "<E>":
                break
        return ' '.join(tokens)


class BigramModel(NGramModelBase):
    def get_sentence_probability(self, sentence: str):
        tokens = self.tokenize_sentence(sentence)
        probability = 0
        probability += self.unigram_smooth(tokens[0])
        for pair in pairwise(tokens):
            probability += self.bigram_smooth(pair[0], pair[1])
        return probability

    def get_model_title(self) -> str:
        return "Bigrams Model"

    def generate_random_sentence(self, length: int) -> str:
        tokens = ["<b>"]
        i = 1
        while i < length:
            prev_token = tokens[i - 1]
            next_token = self.generate_random_bigram_token(prev_token)
            tokens.append(next_token)
            i += 1
            if next_token == "<e>" or next_token == "<E>":
                break
        return ' '.join(tokens)


class TrigramModel(NGramModelBase):

    def get_sentence_probability(self, sentence: str):
        tokens = self.tokenize_sentence(sentence)
        probability = 0
        probability += self.unigram_smooth(tokens[0])
        probability += self.bigram_smooth(tokens[0], tokens[1])
        for triplet in triplewise(tokens):
            probability += self.trigram_interpolation(triplet[0], triplet[1], triplet[2])
        return probability

    def get_model_title(self) -> str:
        return "Trigrams Model"

    def generate_random_sentence(self, length: int) -> str:
        tokens = ["<b>", self.generate_random_bigram_token("<B>")]
        i = 2
        while i < length:
            prev_tokens = (tokens[i - 2].lower(), tokens[i - 1].lower())
            next_token = self.generate_random_trigram_token(prev_tokens)
            tokens.append(next_token)
            i += 1
            if next_token == "<e>" or next_token == "<E>":
                break
        return ' '.join(tokens)

    def trigram_interpolation(self, token_1, token_2, token_3):
        param1 = param2 = param3 = 0.3333
        unigram_prob = self.unigram_smooth(token_3)
        bigram_prob = self.bigram_smooth(token_2, token_3)
        trigram_prob = self.trigram_smooth(token_1, token_2, token_3)
        return param1 * unigram_prob + param2 * bigram_prob + param3 * trigram_prob


class NGramModel:

    def __init__(self, corpus: Corpus):
        self.unigram_model = UnigramModel(corpus)
        self.bigram_model = BigramModel(corpus)
        self.trigram_model = TrigramModel(corpus)
        self.corpus = corpus
        return

    def get_sentences_probability(self, sentences: list) -> str:
        result = ""
        result += "*** Sentence Predictions ***\n\n"
        result += self.unigram_model.get_sentences_probability(sentences)
        result += self.bigram_model.get_sentences_probability(sentences)
        result += self.trigram_model.get_sentences_probability(sentences)
        return result

    def get_random_sentences(self) -> str:
        result = ""
        corpus.update_length_probabilities()
        length = \
            choices(list(self.corpus.lengths_probabilities.keys()), weights=list(self.corpus.lengths_probabilities.values()), k=1)[0]
        result += self.unigram_model.get_random_sentences(length)
        result += "\n"
        result += self.bigram_model.get_random_sentences(length)
        result += "\n"
        result += self.trigram_model.get_random_sentences(length)
        result += "\n"
        return result


if __name__ == '__main__':
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path
    start_time = process_time()
    corpus = Corpus()

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            corpus.add_xml_file_to_corpus(os.path.join(xml_dir, xml_file))
    print("Corpus building completed")

    ngram = NGramModel(corpus)

    sentences = ['May the Force be with you.', 'I’m going to make him an offer he can’t refuse.',
                 'Ogres are like onions.', 'You’re tearing me apart, Lisa!', 'I live my life one quarter at a time.']

    ngram.get_sentences_probability(sentences)
    # second task - add the required tokens to the pre-existing corpus
    corpus.add_begin_and_end_tokens()
    try:
        with open(output_file, encoding="utf8", mode="w") as file:
            # print the first task's output
            file.write(ngram.get_sentences_probability(sentences))
            # print the second task's output
            file.write(ngram.get_random_sentences())
            print(f"Successfully written to file: {output_file}")
    except:
        print("Failed writing to file!")
    print(f"Elapsed Time: {process_time() - start_time}")
