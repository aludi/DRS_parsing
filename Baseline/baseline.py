#!/usr/bin/env python


import nltk
from nltk.corpus import wordnet as wn


def read_data():
    '''Reads in the data from (currently) only the development file
    and returns this as a list. Pops the last element, because it is empty.'''
    with open('../PMB/parsing/layer_data/4.0.0/en/gold/dev.conll') as file:
        data = file.read()
        data = data.split('\n\n')
        data.pop(-1)
    return data


def extract_features(data):
    '''Extracts specific features from the data, such as tokens and possibly other
    things. Extracting extra features can be done by adding them in the loops.'''
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
    '''Conducts part-of-speech tagging on the extracted tokens from
    the data using NLTK.'''
    pos_tags = []
    for sentence in tokens:
        pos_tags.append(nltk.pos_tag(sentence))
    return pos_tags


def lookup_wn(pos_tags):
    '''Looks up the tokens on WordNet and annotates it with the first result
    that is found. Returns a list of tuples with the word and the WordNet result.'''
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


def main():
    data = read_data()
    tokens, roles = extract_features(data)
    pos_tags = pos_tag(tokens)
    wn_annotations = lookup_wn(pos_tags)


if __name__ == "__main__":
    main()