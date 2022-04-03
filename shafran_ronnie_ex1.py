import os
import re
import xml.etree.ElementTree as ET
from sys import argv
from string import punctuation
from time import process_time


class Token:

    def __init__(self, tag, c5, hw, pos, text):
        self.pos = pos
        self.c5 = c5
        self.hw = hw
        self.text = text
        self.tag = tag


class Sentence:

    def __init__(self, index):
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


class Corpus:

    def __init__(self):
        self.documents = []
        self.sentences = []
        self.num_of_tokens = 0
        self.sentence_index = 1
        self.distinct_words = dict()

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
                if token.tag == 'w':
                    self.add_word_to_distinct_dict(token.text)
        self.add_sentence(sentence)
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
            corpus.add_sentence_from_xml_element(sentence)
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


if __name__ == '__main__':

    start_time = process_time()

    xml_dir = argv[1]  # directory containing xml files from the BNC corpus (not a zip file)
    wiki_dir = argv[2]  # directory containing text files from Wikipedia (not a zip file)
    output_file = argv[3]  # output file path

    corpus = Corpus()

    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            corpus.add_xml_file_to_corpus(os.path.join(xml_dir, xml_file))

    for wiki_file in os.listdir(wiki_dir):
        if wiki_file.endswith(".txt"):
            corpus.add_text_file_to_corpus(os.path.join(wiki_dir, wiki_file))

    corpus.create_text_file(output_file)
    print(f"Time Elapsed: {process_time() - start_time}")
