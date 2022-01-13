#!/usr/bin/env python


import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset
from nltk.corpus.reader.wordnet import WordNetError
from collections import defaultdict
import numpy as np
import regex as re
import time
from itertools import permutations

'''
Ti di:
- clean up code, reogrnaize and put into different file
- fix accuracy metric
- (new baseline)
- look at specific examples of stuff that goes wrong (when are synsets not generated)
'''


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


def lookup_wn(pos_tags, num_synsets):
    """Looks up the tokens on WordNet and annotates it with the first result
    that is found. Returns a list of tuples with the word and the WordNet result."""
    # To-do: incorporate named entities as fixed WordNet results, for
    # example Celestial-Seasonings (organization) as company.n.01.
    # regex against expcial characters
    pattern = r'[^A-Za-z0-9]+'

    nouns = ['NN', 'NNP', 'NNS', 'PRP', 'WP', 'IN']
    adjectives = ['JJ', 'JJR', 'JJS']
    adverbs = ['RB', 'RBR', 'RBS']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    possible_tags = ['NN', 'NNP', 'NNS', 'PRP', 'WP', 'IN', 'JJ', 'JJR', 'JJS',
                     'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG',
                     'VBN', 'VBP', 'VBZ', 'POS', 'DT', 'CD', 'CC', 'PDT', ]

    wn_annotations = []


    # hand-coded exceptions: 'time' in DRS is always time.n.08
    #print(pos_tags)
    [[(a, b)]] = pos_tags
    b = re.sub(pattern, '', b)
    if a == 'time':
        return [('time', [wn.synset('fourth_dimension.n.01')])]

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
        sentence = lines[0].strip("%")
        temp = [[], [], [], [], [], []]
        full_drs = [sentence]
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
    correct_base_count = 0
    correct_pos_count = 0

    for drs in data:
        #print()
        clauses_drs = drs[1:]
        tagged = pos_tag([drs[0].split(" ")])
        for clause in clauses_drs:
            #print(clause)
            relevant_tag = 0
            relevant_word = 0
            for item in tagged[0]:
                word, tag = item
                if clause[-2] == word:  # match drs word with pos tag
                    relevant_tag = tag

            if clause[1].islower(): # relevant entities are always all lowercase.
                total_count += 1
                clause_name = clause[1] + "." + clause[2].strip('"')

                [(word, baseline_guess)] = lookup_wn([[(clause[1], relevant_tag)]], "max")
                print(word)
                print(word, baseline_guess)
                print("\t", len(baseline_guess))
                ### pos tag baseline
                if len((baseline_guess)) > 0:
                    baseline_pos_guess = baseline_guess[0].name()  # always take the first one for baseline
                    if clause_name == baseline_pos_guess:
                        correct_pos_count += 1


                ### non-pos tag baseline
                list_of_possible_meanings = wn.synsets(clause[1])
                #print("\t", clause[1])
                #print(list_of_possible_meanings)
                print("\t", len(list_of_possible_meanings) - len(baseline_guess), len(baseline_guess), len(list_of_possible_meanings))


                if len(list_of_possible_meanings) > 0:
                    baseline_guess = list_of_possible_meanings[0].name()  # always take the first one for baseline
                    if clause_name == baseline_guess:
                        correct_base_count += 1
        print()
        print("new sentence")


    print(f"baseline correctness on drs: {(correct_base_count/total_count)*100}")
    print(f"baseline + pos-tagged correctness on drs: {(correct_pos_count/total_count)*100}")



def determining_the_gold_similarity_distance(data):

    '''
    If we want to use similarity, we need to know the (order of magnitude) similarity score
    for the gold parse - so if all the labels are correct, what is the similarity score of
    the sentence as a whole. This function calculates that by doing a triangular matrix calculation,
    for every synset (in the drs), it does this
    sim ... s1 ... s2... s3 ...
    s1 .... 1 ... 0.5 ...0.7...
    s2 ............1.....0.9...
    s3 .................. 1...
    ...
    with list comprehension and the similarity function built into wordnet.
    Then, it sums all the similarity scores (not including the similarity of s1/s1, s2/s2, etc.
    And normalises it by dividing by drs length

    And then it calculates some relevant stats over the collection of drs's. For now, that
    looks like this:
    stats on gold (using wu-palmer similarity... not sure why tbh):
	mean 0.44161341864681675
	median 0.38295454545454544
	standard dev 0.2431369914565437
	variance 0.0591155966145394
	So the best gold parse still only has a mean similarity score of 0.23...

    :param data:
    :return:
    '''
    total_count = 0
    statistics = []
    for drs in data:
        clauses_drs = drs[1:]
        relevant_synsets = []
        for clause in clauses_drs:  # this part gets the synset from the relevant words

            if clause[1].islower():  # relevant entities are always all lowercase.
                total_count += 1
                clause_name = clause[1] + "." + clause[2].strip('"')

                try:
                    relevant_synsets.append(wn.synset(str(clause_name)))
                except WordNetError: # this means that the gold parse is wrong! (nltk.corpus.reader.wordnet.WordNetError: no lemma 'state' with part of speech 'a')
                    try:
                        relevant_synsets.append(wn.synsets(clause[1])[0])    # select the first appropriate one
                    except IndexError:
                        # this word does not exist in wordnet
                        pass

        sum_of_similarity = 0   # this part calculates similarity
        for i in range(0, len(relevant_synsets)-1):  # a 1-d array of words
            for j in range(i+1, len(relevant_synsets)):
                sum_of_similarity += relevant_synsets[i].wup_similarity(relevant_synsets[j])
        statistics.append(sum_of_similarity/len(relevant_synsets))
        #print("normalised similarity for gold parse", sum_of_similarity/len(relevant_synsets))

    print("stats on gold")
    print("\tmean", np.mean(statistics))
    print("\tmedian", np.median(statistics))
    print("\tstandard dev", np.std(statistics))
    print("\tvariance", np.var(statistics))



def test_similarity_matrix(data):
    '''

    for now, on the first fifty drs' (otherwise its too slow):
    - 0th step, get the gold tags for  the drs and put them aside

    - for every word in the drs, look for its collection of synsets (that fit with its pos-tag sometimes)
    - this results in a list [w1, w2, ..., wn], where n = number of words with synsets in the drs, and w1 = collection of synsets for word 1, etc.
    - find the relevant similarity comparisons - we only need to compare the upper triangle (as in def determining_gold...)
         - for a drs with 3 wordnet-words, we only need similarity W1/W2, W1/W3 and W2/W3
    - then, we do matrix multiplication to get all relevant similarities:
        - if w1 = [p1, p2... pm] synsets, and w2 = [q1, q2, ..., qm] synsets, then we do w1Xw2 with similarity (so w1.similarity(w2) for all x in w1 and y in w2
        - we do this for all pair-wise sets (so for a drs with 3 words,  w1/w2, w1/w3 and w2/w3, where w1 etc. are now vectors instead of scalars.
    - so now we have similarity scores for all possible combinations in the sentence.
    - Now, we assume a given meaning (and assign w1 = p1, w2 = q1, etc.), and collect and sum the similarity scores for this sentence complex
    - Then we divide by the number of words for normalization
    - We do this for every combination of assigned meanings, and so we get an overall similarity score for a given sentence meaning
    - and then we pick a best and worse one and compare the best one to the gold

    If we set the pos tags we get an accuracy of like,
    36%.

    Without setting pos tags:
    mean 0.3610952380952381
	median 0.3333333333333333
	standard dev 0.23795834636165758
	variance 0.05662417460317459

	With some hand-made adjustments (always tagging time as time.n.08, and adding RPR tag to list
	performance on matching similar-based tags to gold parse
	mean 0.727
	median 0.67
	standard dev 0.2077041164734103
	variance 0.043141

    :param data:
    :return:
    '''
    total_count = 0
    statistics = []
    skip_counter = 0
    runtime_start = time.time()
    complexity_threshold = 1000
    for drs in data:
        clauses_drs = drs[1:]
        relevant_synsets = []
        tagged = pos_tag([drs[0].split(" ")])

        gold_tags = []

        for clause in clauses_drs:  # this part gets the synset from the relevant words
            relevant_tag = 0
            #print(clause)
            for item in tagged[0]:
                word, tag = item
                if clause[-2] == word:  # match drs word with pos tag
                    relevant_tag = tag

            if clause[1].islower() and clause[1].isalnum():  # relevant entities are always all lowercase and don't contain weird signs.
                total_count += 1
                clause_name = clause[1] + "." + clause[2].strip('"') # this is the gold tags
                try:
                    gold_tags.append(wn.synset(clause_name)) # should always work...?
                except WordNetError:
                    print("whole clause ", clause)

                    print("clause ", clause[1])
                    print(wn.synsets(clause[1]))
                    try:
                        found_synsets = wn.synsets(clause[1])
                        if found_synsets != []:
                            gold_tags.append(wn.synsets(clause[1])[0])
                    except IndexError:
                        gold_tags = gold_tags
                        print("strange token error, no synsets found")
                        pass
                #collected_synsets = wn.synsets(clause[1])
                #print(collected_synsets)
                #print(clause[1], relevant_tag)
                [(word, collected_synsets)] = lookup_wn([[(clause[1], relevant_tag)]], "max")
                try:
                    if collected_synsets != '' and collected_synsets != []:
                        relevant_synsets.append(collected_synsets)
                except WordNetError:  # this means that the gold parse is wrong! (nltk.corpus.reader.wordnet.WordNetError: no lemma 'state' with part of speech 'a')
                    try:
                        found_synsets = wn.synsets(clause[1])
                        if found_synsets != []:
                            relevant_synsets.append(wn.synsets(clause[1]))  # select the first appropriate one
                    except IndexError:
                        print("strange token error, no synsets found")
                        # this word does not exist in wordnet
                        pass


        sentence = relevant_synsets
        # base test, only take c3 and c4:
        # len = number of words
        # this first loop determines which things we have to multiply with each other

        relevant_sims = []

        for i in range(0, len(sentence)-1):  # a 1-d array of words
            for j in range(i + 1, len(sentence)):
                relevant_sims.append((i, j))
        d = defaultdict(dict)


        for cal in relevant_sims:
            i, j = cal
            synsets1, synsets2 = sentence[i], sentence[j]
            #print(synsets1, synsets2)
            # now find similarity between all these synsets
            for x in synsets1:
                for y in synsets2:
                    #print(x, y)
                    #print(x.wup_similarity(y))
                    d[x][y] = x.wup_similarity(y)

        # then determine the meaning per sentence for given word labels


        #print(relevant_sims)
        l = []
        size = 1
        for list_of_sets in sentence:
            l.append(len(list_of_sets))
            size *= len(list_of_sets)
        # the following is very slow because it is nested for-loops :/
        sum_similarity_dict = {}  # i HATE numpy
        sum_similarity = np.arange(size).reshape(l)
        #print(sum_similarity)

        if size > complexity_threshold:    # computer says no to too many options, need a way to narrow it down
            #print("complex too long for computer.. take out during final")
            skip_counter += 1
            continue

        with np.nditer(sum_similarity, flags=['multi_index'], op_flags=['readwrite']) as it:
            for x in it:
                r = it.multi_index
                sum_score = 0
                list_sense_meanings = []
                for list_index in range(0, len(sentence)):
                    #print("\t", sentence[list_index][it.multi_index[list_index]])
                    list_sense_meanings.append(sentence[list_index][it.multi_index[list_index]])
                for p in relevant_sims:
                    x, y = p
                    # x is what synset should be in first position
                    # y is what synset should be in second position
                    # print("senses", sentence[x][r[x]], sentence[y][r[y]])
                    # print(d[sentence[x][r[x]]][sentence[y][r[y]]])
                    sum_score += d[sentence[x][r[x]]][sentence[y][r[y]]]    # add some pruning because otherwise it takes atrociously long
                #print("average similarity score: ", sum_score / len(relevant_sims))
                try:
                    sum_similarity_dict[tuple(list_sense_meanings)] = sum_score / len(relevant_sims)
                except ZeroDivisionError:
                    sum_similarity_dict[tuple(list_sense_meanings)] = 0

        #check
        #for x in sum_similarity_dict.keys():
        #    print(x, sum_similarity_dict[x])

        accuracy = calculate_accuracy(gold_tags, max(sum_similarity_dict.items(), key=lambda k: k[1])[0])

        y = set(max(sum_similarity_dict.items(), key=lambda k: k[1])[0])
        x = set(gold_tags)

        print("GOLD", gold_tags)
        print("best", max(sum_similarity_dict.items(), key=lambda k: k[1]))     # todo add a progress bar
        #print("worst", min(sum_similarity_dict.items(), key=lambda k: k[1]))


        # print("\tcompare best to gold tags")
        #print("\t overlap in wordnet labels... ", x & y) # not entirely correct -order matters (and no homonyms)
        print("\t accuracy... ", round(accuracy, 2))# not entirely correct -order matters (and no homonyms)
        print()


        statistics.append(accuracy)    # not the best

    print("PROGRAM PERFORMANCE:")
    print(f"\tnumber of skipped items {skip_counter}, with a threshold of {complexity_threshold}")
    print(f"\truntime: {round(time.time() - runtime_start, 2)} seconds")
    #print(statistics)
    print("MODEL PERFORMANCE:")
    print("performance on matching similar-based tags to gold parse")
    print("\tmean", np.mean(statistics))
    print("\tmedian", np.median(statistics))
    print("\tstandard dev", np.std(statistics))
    print("\tvariance", np.var(statistics))

def calculate_accuracy(gold, found):
    count = 0
    if len(gold) == len(found):
        for i in range(0, len(gold)):
            if gold[i] == found[i]:
                count += 1
        return count/len(gold)
    else:
        print("WRONG SIZE", "gold length ", len(gold), "found length ", len(found))
        print("GOLD", gold)
        print("best", found)  # todo add a progress bar
        return 0








def main():
    download_nltk_packages()
    data, drs_data = read_all_data()
    drs, claused_drs = extract_drs_features(drs_data)
    test_similarity_matrix(claused_drs)
    #baseline_drs_wn(claused_drs)
    #determining_the_gold_similarity_distance(claused_drs)

    #tokens, roles = extract_features(data)
    #pos_tags = pos_tag(tokens)
    #wn_annotations = lookup_wn(pos_tags, 1)


if __name__ == "__main__":
    main()