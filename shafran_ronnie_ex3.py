from sys import argv
import os
import re
import xml.etree.ElementTree as ET
from sys import argv
from string import punctuation
from random import choice
from time import process_time, strftime, gmtime

import numpy as np
from gender_guesser.detector import Detector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier



class Token:

    def __init__(self, tag, c5, hw, pos, text):
        self.pos = pos
        self.c5 = c5
        self.hw = hw
        self.text = text
        self.tag = tag


class Sentence:

    def __init__(self, index: int, authors: list, gender: str):
        self.index = index
        self.tokens = []
        self.length = 0
        self.num_of_tokens = 0
        self.authors = authors
        self.gender = gender

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


class Corpus:

    def __init__(self):
        self.documents = []
        self.sentences = []
        self.num_of_tokens = 0
        self.sentence_index = 1
        self.distinct_words = dict()
        self.male_chunks = []
        self.female_chunks = []

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

    def add_sentence(self, sentence: Sentence):
        self.sentences.append(sentence)
        self.num_of_tokens += sentence.num_of_tokens

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

    def add_sentence_from_xml_element(self, element: ET.Element, authors: list, gender: str) -> Sentence:
        sentence = Sentence(self.sentence_index, authors, gender)
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
                if token.tag == 'w':
                    self.add_word_to_distinct_dict(token.text)
        self.add_sentence(sentence)
        self.sentence_index += 1
        return sentence

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
        authors = self.get_authors(tree)
        gender = self.get_author_gender(authors)
        fill_chunks = gender == 'male' or gender == 'female'
        chunk = Chunk()
        for sentence_tag in root.iter('s'):
            sentence = self.add_sentence_from_xml_element(sentence_tag, authors, gender)
            if fill_chunks:
                if len(chunk) == 10:
                    if gender == 'male':
                        self.male_chunks.append(chunk)
                    else:
                        self.female_chunks.append(chunk)
                    chunk = Chunk()
                chunk.add(sentence)
        self.add_document(self.get_xml_document_title(root))

    def get_xml_document_title(self, root) -> str:
        header_tag = root.find('teiHeader')
        description_tag = header_tag.find('fileDesc')
        title_stmt_tag = description_tag.find('titleStmt')
        title_tag = title_stmt_tag.find('title')
        return title_tag.text

    def get_authors(self, root: ET.ElementTree) -> list:
        header_tag = root.find('teiHeader')
        description_tag = header_tag.find('fileDesc')
        source_tag = description_tag.find('sourceDesc')
        bibl_tag = source_tag.find('bibl')
        authors = []
        author_tags = bibl_tag.iter('author')
        for author_tag in author_tags:
            authors.append(author_tag.text)
        return authors if len(authors) > 0 else None

    def get_author_gender(self, authors: list) -> str:
        if authors is None:
            return "Unknown"
        detector = Detector()
        genders = []
        for author_name in authors:
            parts = author_name.split(', ')
            if len(parts) <= 1:
                continue
            first_name = parts[1]
            gender = detector.get_gender(first_name)
            if gender != 'male' and gender != 'female':
                return "Unknown"
            genders.append(gender)
        if len(genders) == 0:
            return "Unknown"
        first_gender = genders[0]
        for gender in genders:
            if gender != first_gender:
                return "Unknown"
        return first_gender

    def add_text_file_to_corpus(self, file_name: str):
        with open(file_name, encoding="utf8") as text_file:
            self.add_document(self.get_text_document_title(file_name))
            text = text_file.read()
            sentences = self.split_text_to_sentences(text)
            for sentence in sentences:
                tokens = self.split_sentence_to_tokens(sentence)
                sentence_to_add = Sentence(self.sentence_index, None, None)
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


class Chunk:
    def __init__(self):
        self.sentences = []

    def add(self, sentence: Sentence):
        self.sentences.append(sentence)

    def __len__(self):
        return len(self.sentences)

    def __str__(self):
        return ' '.join(str(sentence) for sentence in self.sentences)

    def get_author_gender(self):
        return self.sentences[0].gender


# Implement a "Classify" class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class). Make sure that the class contains the relevant fields for
# classification, and the methods in order to complete the tasks:

class ResultsContainer:
    def __init__(self, vectorizer_name = None):
        self.report = None
        self.cv_acc = None
        self.vectorizer = vectorizer_name

    def __str__(self):
        return f'=={self.vectorizer} Classification==\n\nCross Validation Accuracy: {self.cv_acc}%\n\n{self.report}'



class Classify:

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.male_chunks = corpus.male_chunks
        self.orig_male_chunks = len(self.male_chunks)
        self.female_chunks = corpus.female_chunks
        self.orig_female_chunks = len(self.female_chunks)
        self.bow_results = ResultsContainer('BoW')
        self.custom_results = ResultsContainer('Custom Vectorizer')

    def down_sample(self) -> None:
        while len(self.male_chunks) > len(self.female_chunks):
            chunk = choice(self.male_chunks)
            self.male_chunks.remove(chunk)
        while len(self.male_chunks) < len(self.female_chunks):
            chunk = choice(self.female_chunks)
            self.female_chunks.remove(chunk)

    def perform_classification(self):
        self.down_sample()
        chunks_as_text = self.chunks_to_text()
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        y = np.array([1] * len(self.male_chunks) + [0] * len(self.female_chunks))

        # Classify using Bag Of Words:
        x = CountVectorizer().fit_transform(chunks_as_text)
        cv_acc = self.ten_fold_cv(x,y,knn_classifier)
        train_test_report = self.train_test_split(x,y,knn_classifier)
        self.bow_results.cv_acc = cv_acc
        self.bow_results.report = train_test_report

        # Classify using Custom Vector:



    def chunks_to_text(self):
        male_text = [str(chunk) for chunk in self.male_chunks]
        female_text = [str(chunk) for chunk in self.female_chunks]
        return male_text + female_text

    def ten_fold_cv(self, x, y, classifier) -> float:
        cv_score = cross_val_score(classifier, x, y, cv=10)
        return round(cv_score.mean() * 100, 4)

    def train_test_split(self, x, y, classifier):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        labels = ['male', 'female']
        return classification_report(y_test, prediction, target_names=labels)

    def write_results_to_file(self, cv_acc, train_test_report, vectorizer_name, file_path):
        with open(file_path, encoding="utf8", mode="w") as file:
            file.write(str(self))


if __name__ == '__main__':
    xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    output_file = argv[2]  # output file name, full path
    start_time = process_time()
    corpus = Corpus()

    directory = os.listdir(xml_dir)
    for file_num, xml_file in enumerate(directory):
        xml_files_len = len([xml for xml in directory if xml.endswith(".xml")])
        if xml_file.endswith(".xml"):
            corpus.add_xml_file_to_corpus(os.path.join(xml_dir, xml_file))
            if file_num % 100 == 0 :
                print(f'reading XML files: {file_num} files read')
    print('XML file reading complete!')
    classifier = Classify(corpus)
    print('Performing Classification...')
    classifier.perform_classification()
    with open(output_file, encoding="utf8", mode="w") as file:
        file.write(str(classifier.bow_results))
    print(f'Results file created: {output_file}')

    elapsed_time = process_time() - start_time
    print('================')
    print(f'Time Elapsed: {strftime("%H:%M:%S", gmtime(elapsed_time))}')


    # Implement here your program:
    # 1. Create a corpus from the file in the given directory (up to 1000 XML files from the BNC)
    # 2. Create a classification object based on the class implemented above.
    # 3. Classify the chunks of text from the corpus as described in the instructions.
    # 4. Print onto the output file the results from the second task in the wanted format.
