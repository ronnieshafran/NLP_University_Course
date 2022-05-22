from sys import argv
from time import time, strftime, gmtime
from math import log

TAB = "    "
NEW_LINE = '\n'


class Rule:
    def __init__(self, index, source: '', target: [], probability: float):
        self.probability = probability
        self.target = target
        self.source = source
        self.index = index

    def __str__(self):
        return f"{self.index}: {self.source} -> {self.target} ({self.probability})"


class BackChartEntry:
    def __init__(self, probability : float, rule: Rule, left_coords: (), right_coords: ()):
        self.rule = rule
        self.right_coords = right_coords
        self.left_coords = left_coords


class GrammarRules:
    def __init__(self):
        self.rules = []
        self.reverse_rules = {}

    def add_rules_from_file(self, file: ""):
        with open(file) as rules_file:
            for i, line in enumerate(rules_file):
                words = line.strip().split()
                probability = float(words[0])
                source = words[1]
                target = words[3:]
                reversed_key = tuple(target)
                rule = Rule(i, source, reversed_key, probability)
                self.rules.append(rule)
                if self.reverse_rules.get(reversed_key) is None:
                    self.reverse_rules[reversed_key] = []
                self.reverse_rules[reversed_key].append(rule)

    def get_possible_rules(self, targets: ()):
        return self.reverse_rules.get(targets, [])


def cky(sentence: "", grammar: GrammarRules):
    words = [word.strip('\n').strip() for word in sentence.split(' ')]
    n = len(words)
    chart = [[[0 for _ in range(len(grammar.rules))] for _ in range(n + 1)] for _ in range(n)]

    for col in range(1, n + 1):
        word = words[col - 1]
        # init terminals in the main diagonal
        for A in grammar.get_possible_rules((word,)):
            chart[col - 1][col][A.index] = A.probability
        # main loop as in pseudocode
        for row in range(col - 2, -1, -1):
            for k in range(row + 1, col):
                for B in [rule for rule in grammar.rules if chart[row][k][rule.index] > 0]:
                    for C in [rule for rule in grammar.rules if chart[k][col][rule.index] > 0]:
                        for rule in grammar.get_possible_rules((B.source, C.source)):
                            p = rule.probability * chart[row][k][B.index] * chart[k][col][C.index]
                            if p > chart[row][col][rule.index]:
                                chart[row][col][rule.index] = p
    result = chart[0][n][0]
    if result > 0:
        return "", log(result)
    return '*** This sentence is not a member of the language generated by the grammar ***', None


def get_parsing_tree(chart, n):
    return "later"


if __name__ == '__main__':
    input_grammar = argv[1]  # The name of the file that contains the probabilistic grammar
    input_sentences = argv[2]  # The name of the file that contains the input sentences (tests)
    output_trees = argv[3]  # The name of the output file
    start_time = time()

    grammar = GrammarRules()
    print('Adding rules to grammar...')
    grammar.add_rules_from_file(input_grammar)
    print('Performing Probabilistic CKY Algorithm...')
    result = []
    with open(input_sentences) as file:
        for sentence in file:
            result.append(f'Sentence: {sentence}')
            parsing_tree, probability = cky(sentence, grammar)
            if parsing_tree is None:
                result.append("*** This sentence is not a member of the language generated by the grammar ***")
            else:
                result.append(f"Parsing:\n {parsing_tree}\n")
                result.append(f"Log Probability: {probability}\n\n")
    result_str = '\n'.join(result)
    print(result_str)
    # with open(output_trees, 'w', encoding='utf8') as output_file:
    #     output_file.write()
    print(f'Successfully written to file: {output_trees}')
    elapsed_time = time() - start_time
    print('================')
    print(f'Time Elapsed: {strftime("%H:%M:%S", gmtime(elapsed_time))}')
