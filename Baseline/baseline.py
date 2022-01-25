#!/usr/bin/env python

import re

import nltk
from nltk.corpus import wordnet as wn


def download_nltk_packages():
    """Downloads the NLTK packages necessary for this project."""
    nltk.download('wordnet', quiet=True)


def read_drs_data(file):
    """Reads in the data from the DRS-clause structure file provided
    and returns it as a list."""
    with open(file, encoding='utf8') as file:
        data = file.read()
        data = data.split('\n\n')
        data.pop(-1)
    return data


def extract_drs_features(data):
    """Converts the DRS clause data into a more usable form."""
    structured_data = []
    claused_drs = []
    for item in data:
        lines = item.split('\n')
        lines = lines[2:-1]          # remove preamble
        sentence = lines[0].strip("%")
        temp = [[], [], [], [], [], []]
        full_drs = [sentence]
        for line in lines:
            clause = []
            line = line.split(' ')
            if "%" in line[0]:          # useless phrase 'also', "ah!"
                continue                # skip
            temp[0].append(line[0])     # The first argument of a clause is always a variable for a DRS.
            temp[1].append(line[1])     # The second argument determines the type of the clause.
            temp[2].append(line[2])     # The third and fourth argument are always variables
            if line[3] != '':           # or constants (constants are enclosed in double quotes).
                temp[3].append(line[3])
            temp[4].append(line[-2])    # add words
            temp[5].append(line[-1])    # and index in sentence for bookkeeping
            if line[3] != '':           # could be more efficient
                clause = [line[0], line[1], line[2], line[3], line[-2], line[-1]]
            else:
                clause = [line[0], line[1], line[2], line[-2], line[-1]]
            full_drs.append(clause)
        structured_data.append(temp)
        claused_drs.append(full_drs)
    return structured_data, claused_drs


def pos_tag(tokens):
    """Conducts part-of-speech tagging on the extracted tokens from
    the data using NLTK."""
    pos_tags = []
    for sentence in tokens:
        pos_tags.append(nltk.pos_tag(sentence))
    return pos_tags


def lookup_wn(pos_tags, num_synsets):
    """Looks up the tokens on WordNet and annotates it with the first result
    that is found. Returns a list of tuples with the word and the WordNet result."""
    # To-do: incorporate named entities as fixed WordNet results, for
    # example Celestial-Seasonings (organization) as company.n.01.
    # Regex against expcial characters
    pattern = r'[^A-Za-z0-9]+'
    nouns = ['NN', 'NNP', 'NNS', 'PRP', 'WP', 'IN', "NNPS"]
    adjectives = ['JJ', 'JJR', 'JJS']
    adverbs = ['RB', 'RBR', 'RBS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    possible_tags = ['NN', 'NNP', 'NNS', "NNPS", 'PRP', 'WP', 'IN', 'JJ', 'JJR', 'JJS',
                     'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG',
                     'VBN', 'VBP', 'VBZ', 'POS', 'DT', 'CD', 'CC', 'PDT', 'WDT', 'WRB', "$", "RP"]
    wn_annotations = []

    # hand-coded exceptions: 'time' in DRS is always time.n.08
    [[(a, b)]] = pos_tags
    b = re.sub(pattern, '', b)
    if a == 'time':
        return [('time', [wn.synset('fourth_dimension.n.01')])]
    if a == 'entity':
        return [('entity', [wn.synset('entity.n.01')])]
    if a == 'event':
        return [('event', [wn.synset('event.n.01')])]
    if a == 'person':
        return [('person', [wn.synset('person.n.01')])]

    for sentence in pos_tags:
        for word in sentence:
            suggested_pos_tag = re.sub(pattern, '', word[1])
            if suggested_pos_tag in possible_tags:
                wn_pos = ''
                if suggested_pos_tag in nouns:
                    wn_pos = wn.NOUN
                elif suggested_pos_tag in adjectives:
                    wn_pos = wn.ADJ
                elif suggested_pos_tag in adverbs:
                    wn_pos = wn.ADV
                elif suggested_pos_tag in verbs:
                    wn_pos = wn.VERB

                if wn.synsets(word[0], pos=wn_pos) != []:
                    if num_synsets == "max":
                        wn_annotations.append((word[0], wn.synsets(word[0], pos=wn_pos)))
                    elif num_synsets:
                        wn_annotations.append((word[0], wn.synsets(word[0], pos=wn_pos)[:num_synsets]))

                else:
                    wn_annotations.append((word[0], wn.synsets(word[0]))) # don't filter for pos tags, just shoot in the dark
            else:
                wn_annotations.append((word[0], ''))
    #print(wn_annotations)
    return wn_annotations


def calculate_frequency(claused_drs):
    '''Calculates how often each WordNet token (synset label) occurs for each word
    in the dev and test set. Returns a dictionary of dictionaries with the word
    and the occurrence of each WordNet token for that word.'''
    frequency_counts = {}
    for drs in claused_drs:
        annotations = drs[1:]
        for clause in annotations:
            if len(clause[2]) != 2:
                word = clause[1]
                token = clause[2]
                if word not in frequency_counts:
                    frequency_counts[word] = {token: 1}
                else:
                    if token not in frequency_counts[word]:
                        frequency_counts[word][token] = 1
                    else:
                        frequency_counts[word][token] += 1
    return frequency_counts


def determine_highest(frequency_counts):
    '''Determines what the most frequent WordNet token is for the words
    in the dev and test set and returns that.'''
    most_frequent = []
    for key, value in frequency_counts.items():
        if len(value) >= 2:
            highest_token = ""
            highest_value = 0
            for key2, value2 in value.items():
                if value2 >= highest_value:
                    highest_token = key2
                    highest_value = value2
            most_frequent.append((key, highest_token))
        else:
            most_frequent.append((key, list(value)[0]))
    return most_frequent


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
    correct_base_count = 0
    correct_pos_count = 0

    for drs in data:
        clauses_drs = drs[1:]
        #print(drs[0])
        tagged = pos_tag([drs[0].split(" ")[1:]])
        for clause in clauses_drs:
            relevant_tag = 0
            relevant_word = 0
            for item in tagged[0]:
                word, tag = item
                if clause[-2] == word:  # match drs word with pos tag
                    relevant_tag = tag
                else:
                    relevant_tag = 'NN' # otherwise just make it NN, since we can't find it

            if clause[1].islower(): # relevant entities are always all lowercase.
                total_count += 1
                clause_name = clause[1] + "." + clause[2].strip('"')
                #print(clause[1], relevant_tag)
                [(word, baseline_guess)] = lookup_wn([[(clause[1], relevant_tag)]], "max")
                #print(word)
                #print(word, baseline_guess)
                #print("\t", len(baseline_guess))
                ### pos tag baseline
                if len((baseline_guess)) > 0:
                    #print(baseline_guess)
                    baseline_pos_guess = baseline_guess[0].name()  # always take the first one for baseline
                    if clause_name == baseline_pos_guess:
                        correct_pos_count += 1


                ### non-pos tag baseline
                list_of_possible_meanings = wn.synsets(clause[1])
                #print("\t", clause[1])
                #print(list_of_possible_meanings)
                #print("\t", len(list_of_possible_meanings) - len(baseline_guess), len(baseline_guess), len(list_of_possible_meanings))


                if len(list_of_possible_meanings) > 0:
                    baseline_guess = list_of_possible_meanings[0].name()  # always take the first one for baseline
                    if clause_name == baseline_guess:
                        correct_base_count += 1
        #print()
        #print("-----new sentence------")
    print(f"baseline correctness on drs: {(correct_base_count/total_count)*100}")
    #print(f"baseline + pos-tagged correctness on drs: {(correct_pos_count/total_count)*100}")


def baseline_most_frequent(most_frequent, data):
    '''Baseline system that gets most frequent synsets on the train and dev set, evaluates
    on the test set. Does not use the hard-coded synsets, because it figures these out on
    its own with the frequencies.

    Baseline accuracy: 86.6% (?)'''
    total_count = 0
    correct_base_count = 0
    correct_pos_count = 0
    for drs in data:
        #print(drs)
        clauses_drs = drs[1:]
        tagged = pos_tag([drs[0].split(" ")[1:]])
        for clause in clauses_drs:
            baseline_guess = ''
            for item in tagged[0]:
                word, tag = item
                if clause[-2] == word:
                    relevant_tag = tag
            if clause[1].islower():
                total_count += 1
                clause_name = clause[1] + "." + clause[2].strip('"')
                #print(clause[1], relevant_tag)
                baseline_guess = ''
                for token, synset in most_frequent:
                    if clause[1] == token:
                        #print(token, synset, clause[1])
                        baseline_guess = clause[1] + '.' + synset.strip('"')
                if baseline_guess == '':
                    [(word, baseline_guess)] = lookup_wn([[(clause[1], relevant_tag)]], "max")
                    if len((baseline_guess)) > 0:
                        #print(baseline_guess)
                        baseline_pos_guess = baseline_guess[0].name()  # always take the first one for baseline
                        if clause_name == baseline_pos_guess:
                            correct_pos_count += 1

                    ### non-pos tag baseline
                    list_of_possible_meanings = wn.synsets(clause[1])
                    # print("\t", clause[1])
                    # print(list_of_possible_meanings)
                    #print("\t", len(list_of_possible_meanings) - len(baseline_guess), len(baseline_guess),
                          #len(list_of_possible_meanings))

                    if len(list_of_possible_meanings) > 0:
                        baseline_guess = list_of_possible_meanings[0].name()  # always take the first one for baseline
                if clause_name == baseline_guess:
                    #print(clause_name, baseline_guess)
                    correct_base_count += 1
    print(f"baseline correctness on drs: {(correct_base_count/total_count)*100}")


def main():
    download_nltk_packages()
    train = read_drs_data('../PMB/data/pmb-4.0.0/gold/train.txt')
    dev = read_drs_data('../PMB/data/pmb-4.0.0/gold/dev.txt')
    test = read_drs_data('../PMB/data/pmb-4.0.0/gold/test.txt')

    structured_drs, claused_drs = extract_drs_features(train + dev)
    structured_drs_test, claused_drs_test = extract_drs_features(test)
    frequency_counts = calculate_frequency(claused_drs)
    most_frequent = determine_highest(frequency_counts)

    baseline_most_frequent(most_frequent, claused_drs_test)
    baseline_drs_wn(claused_drs_test)


if __name__ == "__main__":
    main()