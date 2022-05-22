from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sys import argv
import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from string import punctuation
from random import randrange
from more_itertools import pairwise, triplewise
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time, strftime, gmtime


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
                self.tokenize_complex_child(child, sentence)
                for c in child:
                    self.tokenize_complex_child(c, sentence)
                continue
            token = self.tokenize_xml_element(child)
            if token.text is not None:
                sentence.add_token(token)
                sentence.length += len(token.text)
                self.add_word_to_distinct_dict(token.text.lower())
        self.add_sentence(sentence)
        self.add_tokens_to_bigram(sentence.tokens)
        self.add_tokens_to_trigram(sentence.tokens)
        self.sentence_index += 1

    def tokenize_complex_child(self, element, sentence):
        for subelement in element:
            if len(subelement) > 0:
                self.tokenize_complex_child(subelement, sentence)
        token = self.tokenize_xml_element(element)
        if token.text is not None:
            sentence.add_token(token)
            sentence.length += len(token.text)

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


class Tweet:
    def __init__(self, text, category):
        self.text = text
        self.category = category


def split_sentence_to_tokens(sentence: str) -> []:
    initial_split = re.split('(\W)', sentence)
    return [token for token in initial_split if token.strip()]


# Ex 1 a
def similarity_between_pairs(model) -> str:
    pairs = [('good', 'bad'), ('short', 'tall'), ('friend', 'mate'), ('father', 'dad'), ('fight', 'battle'), ('start', 'end'),
             ('hawk', 'eagle'), ('house', 'home'), ('hot', 'cold'), ('software', 'program')]
    results_list = ['Word Pairs And Distances:']
    for i, pair in enumerate(pairs):
        results_list.append(f'{i + 1}. {pair[0]} - {pair[1]} : {model.similarity(pair[0], pair[1])}')
    return '\n'.join(results_list)


# Ex 1 b
def similarity_between_analogies(model) -> str:
    analogies_list = ['\n\nAnalogies:']
    most_similar_list = ['Most Similar:']
    distances_list = ['Distances:']
    analogies = [(('red', 'color'), ('circle', 'shape')),
                 (('sand', 'beach'), ('water', 'ocean')),
                 (('cat', 'feline'), ('dog', 'canine')),
                 (('glove', 'hand'), ('sock', 'foot')),
                 (('cool', 'cold'), ('warm', 'hot'))]
    for i, (analogy_1, analogy_2) in enumerate(analogies):
        expected_word = analogy_2[1]
        actual_word = model.most_similar(positive=[analogy_2[0], analogy_1[1]], negative=[analogy_1[0]])[0][0]
        distance = model.similarity(actual_word, expected_word)
        analogies_list.append(f'{i + 1}. {analogy_1[0]} : {analogy_1[1]} , {analogy_2[0]} : {analogy_2[1]}')
        most_similar_list.append(f'{i + 1}. {analogy_2[0]} + {analogy_1[1]} - {analogy_1[0]} = {actual_word}')
        distances_list.append(f'{i + 1}. {expected_word} - {actual_word} : {distance}')
    results = ['\n'.join(analogies_list), '\n'.join(most_similar_list), '\n'.join(distances_list)]
    return '\n\n'.join(results)


# Generate an alternate version of the lyrics by replacing one word per line
def get_altered_song(model: KeyedVectors, song_file: str, corpus: Corpus):
    words_to_change = [
        'baby',
        'you',
        'at',
        'got',
        'shut',
        'sipping',
        'good',
        'house',
        'like',
        'dancing',
        'wing',
        'mansion',
        'playing',
        'straight',
        'lay',
        'door',
        'door',
        'door',
        'door',
        'way',
        'like',
        'coming',
        'tight',
        'bite',
        'smoke',
        'hungry',
        'keep',
        'love',
        'kissing',
        'bathtub',
        'jump',
        'playing',
        'straight',
        'lay',
        'open',
        'open',
        'open',
        'open',
        'way',
        'want',
        'come',
        'baby',
        'baby',
        'you',
        'ah',
        'door',
        'door',
        'door',
        'way',
        'like',
        'woo',
        'tell',
        'coming',
        'woo',
        'woo',
        'la',
        'coming',
        'waiting',
        'adore',
        'waiting',
        'me',
        'waiting',
        'on',
        'la'
    ]
    with open(song_file) as lyrics:
        final_text = ["\n\n=== New Hit ===\n"]
        for line, word in zip(lyrics, words_to_change):
            line = line.strip('\n')
            tokens = corpus.split_sentence_to_tokens(line)
            similar_words = model.most_similar(word)
            trigram_found = False
            bigram_found = False
            while word in tokens:
                index = tokens.index(word)
                key, list_index = get_trigram_key(index, tokens)
                trigram_found, replacement = GetTrigramCount(corpus, key, list_index, similar_words, trigram_found)
                if not trigram_found:
                    bi_keys, list_indices = GetBigramKey(index, tokens)
                    bigram_found, replacement = GetBigramCount(bi_keys, bigram_found, corpus, list_indices, similar_words)
                if not bigram_found:
                    replacement = similar_words[0][0]
                tokens[index] = replacement
            final_text.append(' '.join(tokens))
        return '\n'.join(final_text)


def GetBigramCount(bi_keys, bigram_found, corpus, list_indices, similar_words):
    max_count = 0
    replacement = ''
    for sim_word, _ in similar_words:
        bi_sum = 0
        for key, list_index in zip(bi_keys, list_indices):
            sim_key = key
            sim_key[list_index] = sim_word
            bi_count = corpus.bigram_dict.get(tuple(sim_key), 0)
            bi_sum += bi_count
            if bi_sum > max_count:
                bigram_found = True
                max_count = bi_count
                replacement = sim_word
    return bigram_found, replacement


def GetBigramKey(index, tokens):
    if index == 0:
        bi_keys = [[tokens[0], tokens[1]]]
        list_indices = [0]
    elif index == len(tokens) - 1:
        bi_keys = [[tokens[-2], tokens[-1]]]
        list_indices = [1]
    else:
        bi_keys = [[tokens[index - 1], tokens[index]], [tokens[index], tokens[index + 1]]]
        list_indices = [1, 0]
    return bi_keys, list_indices


def GetTrigramCount(corpus, key, list_index, similar_words, trigram_found):
    max_count = 0
    replacement = ''
    for sim_word, _ in similar_words:
        sim_key = key
        sim_key[list_index] = sim_word
        tri_count = corpus.trigram_dict.get(tuple(sim_key), 0)
        if tri_count > max_count:
            trigram_found = True
            max_count = tri_count
            replacement = sim_word
    return trigram_found, replacement


def get_trigram_key(index, tokens):
    if index == 0:
        key = [tokens[0], tokens[1], tokens[2]]
        list_index = 0
    elif index == len(tokens) - 1:
        key = [tokens[-3], tokens[-2], tokens[-1]]
        list_index = 2
    else:
        key = [tokens[index - 1], tokens[index], tokens[index + 1]]
        list_index = 1
    return key, list_index


def prepare_kv_file():
    glove2word2vec('glove.6B.50d.txt', 'glove2word2vec.kv')
    pre_trained_model = KeyedVectors.load_word2vec_format('glove2word2vec.kv', binary=False)
    pre_trained_model.save('word2vec_vectors.kv')


# Calculate sum of WiVi
def get_weight_vector(tweet, model, weight_func):
    tweet_tokens = split_sentence_to_tokens(tweet)
    model_length = model.vector_size
    w = [np.full(model_length, weight_func(token)) for token in tweet_tokens]
    v = [np.full(model_length, 1) if token.lower() not in model else model[token.lower()] for token in tweet_tokens]
    result = np.zeros(model_length)
    for weight, vector in zip(w, v):
        result += np.multiply(weight, vector)
    return result / len(tweet_tokens)


def get_tokens_from_category(category: str, tweets: []) -> []:
    tokens = [split_sentence_to_tokens(tweet.text) for tweet in tweets if tweet.category == category]
    return [item for sublist in tokens for item in sublist]


# My custom function, logic explained in the report
def per_category_scores(tweets):
    words_dict = {}
    covid_tokens = get_tokens_from_category('Covid', tweets)
    olympics_tokens = get_tokens_from_category('Olympics', tweets)
    pets_tokens = get_tokens_from_category('Pets', tweets)
    from collections import Counter

    covid_counter = Counter(covid_tokens)
    olympics_counter = Counter(olympics_tokens)
    pets_counter = Counter(pets_tokens)

    covid_unique_keys = [key for key in covid_counter.keys() if key not in olympics_counter.keys() and key not in pets_counter.keys()]
    olympics_unique_keys = [key for key in olympics_counter.keys() if
                            key not in covid_counter.keys() and key not in pets_counter.keys()]
    pets_unique_keys = [key for key in pets_counter.keys() if key not in covid_counter.keys() and key not in olympics_counter.keys()]

    scores_multiplier = {'Covid': 1, 'Olympics': 500, 'Pets': 1000}
    categories = {'Covid': (covid_unique_keys, covid_counter),
                  'Olympics': (olympics_unique_keys, olympics_counter),
                  'Pets': (pets_unique_keys, pets_counter)}

    for category, uniques in categories.items():
        unique_keys = uniques[0]
        unique_counter = uniques[1]
        for unique_key in unique_keys:
            words_dict[unique_key] = unique_counter[unique_key] * scores_multiplier[category]

    return words_dict


def analyze_tweets(model, tweets_file):
    tweets = []
    category = None

    with open(tweets_file, encoding="utf8") as file:
        for line in file:
            if line in ['\n', '\r\n']:
                continue
            if line.startswith('=='):
                category = line.strip().strip('==').strip()
            else:
                tweets.append(Tweet(line.strip(), category))

    custom_words_dict = per_category_scores(tweets)
    weight_functions_dict = {
        'Arithmetic Average': lambda _: 1,
        'Random Score': lambda _: randrange(10),
        'Custom Function': lambda token: custom_words_dict.get(token.lower(), 0)
    }
    pca = PCA(n_components=2)

    for function_name, function in weight_functions_dict.items():
        tweet_weight_vectors = [get_weight_vector(tweet.text, model, function) for tweet in tweets]
        pca.fit(tweet_weight_vectors)
        transform = pca.transform(tweet_weight_vectors)
        plot_pca_results(function_name, transform, tweets)


def plot_pca_results(function_name, transform, tweets):
    plt.title(f'{function_name}: Ronnie Shafran')
    for i, (tweet, (x, y)) in enumerate(zip(tweets, transform)):
        plt.scatter(x, y, s=8, color=get_tweet_color(tweet))
        plt.text(x + .01, y + .01, f'{i} : {tweet.category}', fontsize=8)
    plt.show()


def get_tweet_color(tweet):
    if tweet.category == 'Covid':
        color = 'red'
    elif tweet.category == 'Olympics':
        color = 'pink'
    else:
        color = 'black'
    return color


if __name__ == "__main__":
    kv_file = argv[1]
    xml_dir = argv[2]  # directory containing xml files from the BNC corpus (not a zip file)
    lyrics_file = argv[3]
    tweets_file = argv[4]
    output_file = argv[5]
    start_time = time()
    print('Loading KV File..')
    model = KeyedVectors.load(kv_file, mmap='r')
    print('Building analogies...')
    pairs_sim_string = similarity_between_pairs(model)
    analogies_sim_string = similarity_between_analogies(model)

    corpus = Corpus()
    print('Building corpus...')
    directory = os.listdir(xml_dir)
    for file_num, xml_file in enumerate(directory):
        xml_files_len = len([xml for xml in directory if xml.endswith(".xml")])
        if xml_file.endswith(".xml"):
            corpus.add_xml_file_to_corpus(os.path.join(xml_dir, xml_file))
    print('XML file reading complete!')
    print('Generating new hit...')
    song = get_altered_song(model, lyrics_file, corpus)
    print('Hit generated!')
    print('Analyzing tweets...')
    analyze_tweets(model, tweets_file)
    print('Tweets completed!')
    with open(output_file, encoding="utf8", mode="w") as file:
        file.write(pairs_sim_string)
        file.write(analogies_sim_string)
        file.write(song)
    print(f'Results file created: {output_file}')
    elapsed_time = time() - start_time
    print('================')
    print(f'Time Elapsed: {strftime("%H:%M:%S", gmtime(elapsed_time))}')
