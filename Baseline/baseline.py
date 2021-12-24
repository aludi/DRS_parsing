#!/usr/bin/env python


import nltk
from nltk.corpus import wordnet as wn


def read_data():
    """Reads in the data from (currently) only the development file
    and returns this as a list. Pops the last element, because it is empty."""

    with open('../PMB/parsing/layer_data/4.0.0/en/gold/dev.conll') as file:
        data = file.read()
        data = data.split('\n\n')
        data.pop(-1)
    return data


def read_drs_data():
    """ Reads in the data from the DRS-clause structure dev file
    and returns it as a list."""

    with open('../PMB/data/pmb-4.0.0/gold/dev.txt') as file:
        data = file.read()
        data = data.split('\n\n')
        data.pop(-1)
    return data


def extract_features(data):
    """ Extracts specific features from the data, such as tokens and possibly other
    things. Extracting extra features can be done by adding them in the loops."""

    tokens = []
    roles = []
    for item in data:
        lines = item.split('\n')
        temp = [[], []]
        for line in lines:
            line = line.split('\t')
            if line[0][0] != '#':
                temp[0].append(line[0])
                temp[1].append(line[6])
        tokens.append(temp[0])
        roles.append(temp[1])
    return tokens, roles


def pos_tag(tokens):
    """ Conducts part-of-speech tagging on the extracted tokens from
    the data using NLTK."""

    pos_tags = []
    for sentence in tokens:
        pos_tags.append(nltk.pos_tag(sentence))
    return pos_tags


def lookup_wn(pos_tags):
    """Looks up the tokens on WordNet and annotates it with the first result
    that is found. Returns a list of tuples with the word and the WordNet result."""
    # To-do: incorporate named entities as fixed WordNet results, for
    # example Celestial-Seasonings (organization) as company.n.01.

    nouns =  ['NN', 'NNP', 'NNS']
    adjectives = ['JJ', 'JJR', 'JJS']
    adverbs = ['RB', 'RBR', 'RBS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    possible_tags = ['NN', 'NNP', 'NNS', 'JJ', 'JJR', 'JJS',
                     'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG',
                     'VBN', 'VBP', 'VBZ']
    wn_annotations = []

    for sentence in pos_tags:
        for word in sentence:
            if word[1] in possible_tags:
                wn_pos = ''
                if word[1] in nouns:
                    wn_pos = wn.NOUN
                elif word[1] in adjectives:
                    wn_pos = wn.ADJ
                elif word[1] in adverbs:
                    wn_pos = wn.ADV
                elif word[1] in verbs:
                    wn_pos = wn.VERB

                if wn.synsets(word[0], pos=wn_pos) != []:
                    wn_annotations.append((word[0], wn.synsets(word[0], pos=wn_pos)[0]))
                else:
                    wn_annotations.append((word[0], ''))
            else:
                wn_annotations.append((word[0], ''))

    return wn_annotations


def download_nltk_packages():
    """ Downloads the nltk packages necessary for this project"""
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def read_all_data():
    """ Read the tagged sentences and the drs's into the project"""
    data = read_data()
    drs_data = read_drs_data()
    return data, drs_data


def extract_drs_features(data):
    """ Puts the clause DRS in usable form. """
    structured_data = []
    claused_drs = []
    for item in data:
        lines = item.split('\n')
        lines = lines[2:-1]          # remove preamble
        temp = [[], [], [], [], [], []]
        full_drs = []
        for line in lines:
            clause = []
            line = line.split(' ')
            if "%" in line[0]:       # useless phrase 'also', "ah!"
                continue             # skip
            temp[0].append(line[0])  # The first argument of a clause is always a variable for a DRS.
            temp[1].append(line[1])  # The second argument determines the type of the clause.
            temp[2].append(line[2])  # The third and fourth argument are always variables
            if line[3] != '':        # or constants (constants are enclosed in double quotes).
                temp[3].append(line[3])
            temp[4].append(line[-2])  # add words
            temp[5].append(line[-1])  # and index in sentence for bookkeeping
            if line[3] != '':         # could be more efficient
                clause = [line[0], line[1], line[2], line[3], line[-2], line[-1]]
            else:
                clause = [line[0], line[1], line[2], line[-2], line[-1]]
            full_drs.append(clause)
        structured_data.append(temp)
        claused_drs.append(full_drs)
    return structured_data, claused_drs


def baseline_drs_wn(data):
    """ Primitive baseline system (count-based) for assigning word senses
    to drs-es.
    We take the second item of the DRS. We assume it is a wordnet entry
    if it is in all lowercase (other relations are partially uppercase).
    Then, we get the synsets for that word/relation (not looking at POS tag).
    We pick the first synset, and compare that to the synset given in the gold parse.
    And then we count the number of times that the meaning is correct.

    Current system baseline: 34% correct.
    """
    total_count = 0
    correct_count = 0
    for drs in data:
        for clause in drs:
            if clause[1].islower(): # relevant entities are always all lowercase.
                total_count += 1
                clause_name = clause[1] + "." + clause[2].strip('"')
                list_of_possible_meanings = wn.synsets(clause[1])
                if len(list_of_possible_meanings) > 0:
                    baseline_guess = list_of_possible_meanings[0].name()  # always take the first one for baseline
                    if clause_name == baseline_guess:
                        correct_count += 1

    print(f"baseline correctness on drs: {(correct_count/total_count)*100}")





def main():
    download_nltk_packages()
    data, drs_data = read_all_data()
    drs, claused_drs = extract_drs_features(drs_data)
    baseline_drs_wn(claused_drs)
    tokens, roles = extract_features(data)
    pos_tags = pos_tag(tokens)
    wn_annotations = lookup_wn(pos_tags)


if __name__ == "__main__":
    main()