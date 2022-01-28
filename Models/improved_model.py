#!/usr/bin/env python

import re

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError


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
            # The first argument of a clause is always a variable for a DRS.
            temp[0].append(line[0])
            # The second argument determines the type of the clause.
            temp[1].append(line[1])
            # The third and fourth argument are always variables
            temp[2].append(line[2])
            # or constants (constants are enclosed in double quotes).
            if line[3] != '':
                temp[3].append(line[3])
            temp[4].append(line[-2])    # add words
            temp[5].append(line[-1])    # and index in sentence for bookkeeping
            if line[3] != '':           # could be more efficient
                clause = [line[0], line[1], line[2],
                          line[3], line[-2], line[-1]]
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
                        wn_annotations.append(
                            (word[0], wn.synsets(word[0], pos=wn_pos)))
                    elif num_synsets:
                        wn_annotations.append(
                            (word[0], wn.synsets(word[0], pos=wn_pos)[:num_synsets]))

                else:
                    # don't filter for pos tags, just shoot in the dark
                    wn_annotations.append((word[0], wn.synsets(word[0])))
            else:
                wn_annotations.append((word[0], ''))
    # print(wn_annotations)
    return wn_annotations


def calculate_frequency(claused_drs):
    '''Calculates how often each WordNet token (synset label) occurs for each word
    in the dev and test set. Returns a dictionary of dictionaries with the word
    and the occurrence of each WordNet token for that word.'''
    sense_pattern = re.compile("[a-z]\.[0-9]{2}")
    frequency_counts = {}
    for drs in claused_drs:
        annotations = drs[1:]
        for clause in annotations:
            if len(clause[2]) != 2:
                word = clause[1]
                token = clause[2]
                try:
                    # check that the extracted sense matches the sense pattern and that it exists in wordnet
                    if sense_pattern.match(token.replace('"', '')):
                        wn.synset(word + '.' + token.replace('"', ''))
                        if word not in frequency_counts:
                            frequency_counts[word] = {token: 1}
                        else:
                            if token not in frequency_counts[word]:
                                frequency_counts[word][token] = 1
                            else:
                                frequency_counts[word][token] += 1
                except WordNetError:
                    pass

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


def get_gold_standard(data):
    '''Represents the gold standard as a list of lists of tuples, where tuples are (lemma, gold_synset),
    and a list of tuples represents a sentence.'''
    gold_standard = []
    sense_pattern = re.compile("[a-z]\.[0-9]{2}")
    for drs in data:
        sentence = drs[1:]
        gold_sentence = []
        for clause in sentence:
            if len(clause[2]) != 2:
                word = clause[1]
                sense = clause[2]
                try:
                    # check that the extracted sense matches the sense pattern and that it exists in wordnet
                    if sense_pattern.match(sense.replace('"', '')):
                        wn.synset(word + '.' + sense.replace('"', ''))
                        gold_sentence.append((word, sense.replace('"', '')))
                except WordNetError:
                    pass
        gold_standard.append(gold_sentence)
    return gold_standard


def determine_disambiguation_order(sentence, frequency_counts):
    '''Makes a list of tuples of the form (lemma, #of synsets found in training data), 
    sorted in the ascending oredr of the # of synsets'''
    order = []
    for lemma in sentence:
        if lemma in frequency_counts.keys():
            order.append((lemma, len(frequency_counts[lemma])))
        else:
            try:
                order.append((lemma, len(wn.synsets(lemma))))
                # if lemma =='knock_down':
                #     print(len(wn.synsets(lemma)))
            except WordNetError:
                order.append((lemma, 0))
    return sorted(order, key=lambda x: x[1])


def match_disambiguation_order(gold_standard, frequency_counts):
    '''Since the sentences are disambiguated in a particular order, and the words remain in that order, 
    the gold standard sentences should match that word order for comparison.'''
    new_gold = []
    for sentence in gold_standard:
        # print(sentence)
        order = []
        for lemma in sentence:
            if lemma[0] in frequency_counts.keys():
                order.append(
                    (lemma[0], lemma[1], len(frequency_counts[lemma[0]])))
            else:
                try:
                    order.append(
                        (lemma[0], lemma[1], len(wn.synsets(lemma[0]))))
                except WordNetError:
                    order.append((lemma, 0))
        sort = sorted(order, key=lambda x: x[2])
        # print(sort)
        new_gold.append([(s[0], s[1]) for s in sort])
    return new_gold


def assign_monosemous(disambiguation_order, most_frequent):
    '''Assigns most frequent sense in the training data to lemmas with a single sense.'''
    for idx, item in enumerate(disambiguation_order):
        if item[1] == 1:
            lemma = item[0]
            if list(filter(lambda x: x[0] == lemma, most_frequent)):
                most_freq_sense = list(filter(lambda x: x[0] == lemma, most_frequent))[
                    0][1].replace('"', '')
            else:  # if the word is not in the training data, but it has a single sense in wordnet, assign it
                most_freq_sense = '.'.join(
                    wn.synsets(lemma)[0].name().split('.')[1:])
                # in some cases, word senses correspond to a different word in Wordnet eg, fire_fighter corresponds to fireman.n.04,
                # and doesn't have senses of its own. So, the lemma is changed accordingly to retrieve the correct sense later.
                if lemma != wn.synsets(lemma)[0].name().split('.')[0]:
                    lemma = wn.synsets(lemma)[0].name().split('.')[0]
            disambiguation_order[idx] = (lemma, most_freq_sense)
    return disambiguation_order


def assign_very_frequent_sense(disambiguation_order, frequency_counts):
    '''If the most freqeunt sense makes up more than 90% of the word encounters, then assign it'''
    for idx, item in enumerate(disambiguation_order):
        if item[1] != 0 and item[0] in frequency_counts.keys():
            for sense in frequency_counts[item[0]]:
                if frequency_counts[item[0]][sense] >= 90 * sum(frequency_counts[item[0]].values())/100:
                    disambiguation_order[idx] = (
                        item[0], sense.replace('"', ''))
    return disambiguation_order


def sentence_similarity(sentence, wnsense):
    '''Compute sum of similarity scores between a given sense and the senses of the other words in a sentence.'''
    total_similarity = 0
    for word in sentence:
        if isinstance(word[1], str):  # word is already disambiguated
            sense = wn.synset(word[0] + '.' + word[1])
            total_similarity += wnsense.wup_similarity(sense)
    return total_similarity


def iterative_disambiguation(disambiguation_order, frequency_counts):
    disambiguated = disambiguation_order
    for word in disambiguation_order:

        if isinstance(word[1], int):
            # if word was in the trainig data, retrieve the senses found
            if word[0] in frequency_counts:
                possible_synsets = [wn.synset(
                    word[0] + '.' + sense.replace('"', '')) for sense in frequency_counts[word[0]]]
            else:  # if the word is new, look up its synsets
                if word[1] != 0:
                    possible_synsets = wn.synsets(word[0])
                else:  # there are no entries in WN
                    # remove the word from the sentence
                    disambiguated = list(
                        filter(lambda x: x[0] != word[0], disambiguated))
                    continue
            best_sense = possible_synsets[0]
            best_similarity = 0

            for sense in possible_synsets:
                sent_without_current_word = list(
                    filter(lambda x: x[0] != word[0], disambiguated))
                sent_similarity = sentence_similarity(
                    sent_without_current_word, sense)
                if sent_similarity > best_similarity:
                    best_similarity = sent_similarity
                    best_sense = sense

            # replace the newly found sense in the copied list so that it can be used for similarity in the next iterations.
            idx = disambiguated.index(word)
            disambiguated[idx] = (best_sense.name().split(
                '.')[0], '.'.join(best_sense.name().split('.')[1:]))
    return disambiguated


def disambiguate_sentences(lemma_sentences, frequency_counts, most_frequent):
    '''Sentences are disambigauted step by step as follows:
        1. the disambiguation order is determined based on the polysemy of lemmas
        2. the monosemous words are assigned their senses
        3. words whch were used with a particular sense in x% of the training data are assigned that sense
        4. the rest of words are assigned senses in the disambiguation order based on their similarity 
            with the other words in the sentence whose senses have been assigned already'''
    predictions = []
    for sentence in lemma_sentences:
        disambiguation_order = determine_disambiguation_order(
            sentence, frequency_counts)
        monosemous_assigned = assign_monosemous(
            disambiguation_order, most_frequent)
        very_likely = assign_very_frequent_sense(
            monosemous_assigned, frequency_counts)
        disambiguated = iterative_disambiguation(very_likely, frequency_counts)
        predictions.append(disambiguated)
    return predictions


def get_accuracy(gold_standard, prediction):
    '''Evaluates predictions by computing wsd accuracy, pos accuracy, sentence accuracy.'''

    if len(gold_standard) != len(prediction):
        print("Incompatible datasets. gold = {} pred = {}".format(
            len(gold_standard), len(prediction)))
        return

    total_sentences = 0
    total_senses = 0
    correct_sense_count = 0
    correct_pos_count = 0
    correct_sentences = 0
    sent_data = []
    for gold_sent, pred_sent in zip(gold_standard, prediction):
        if len(gold_sent) != len(pred_sent):
            print("Incompatible sentences. gold = {} pred = {}".format(
                len(gold_sent), len(pred_sent)))
            return
        correct_sent_flag = 1
        total_sentences += 1
        words_in_sent = 0
        correct_words_in_sent = 0
        for gold_word, pred_word in zip(gold_sent, pred_sent):
            total_senses += 1
            words_in_sent += 1
            if gold_word[1] == pred_word[1]:
                correct_sense_count += 1
                correct_pos_count += 1
                correct_words_in_sent += 1
            elif gold_word[1].split('.')[0] == pred_word[1].split('.')[0]:
                correct_pos_count += 1
                correct_sent_flag = 0
            else:
                correct_sent_flag = 0
        # If no word was incorrectly predicted, the sentence is counted as correctly disambiguated
        if correct_sent_flag == 1:
            correct_sentences += 1
        if words_in_sent != 0:
            sent_data.append(correct_words_in_sent/words_in_sent)
    print("WSD Accuracy = {}".format(correct_sense_count/total_senses*100))
    print("POS Accuracy = {}".format(correct_pos_count/total_senses*100))
    print("% of correctly disambiguated sentences = {}".format(
        correct_sentences/total_sentences*100))
    sent_dat = np.array(sent_data)
    print("Mean Accuracy per sentence = {}".format(np.mean(sent_dat)*100))
    print("Median Accuracy per sentence = {}".format(np.median(sent_dat)*100))
    print("Sd Accuracy per sentence = {}".format(np.std(sent_dat)*100))
    print("Variance Accuracy per sentence = {}".format(np.var(sent_dat)*100))


def main():
    download_nltk_packages()
    train = read_drs_data('../PMB/data/pmb-4.0.0/gold/train.txt')
    dev = read_drs_data('../PMB/data/pmb-4.0.0/gold/dev.txt')
    test = read_drs_data('../PMB/data/pmb-4.0.0/gold/test.txt')

    structured_drs, claused_drs = extract_drs_features(train + dev)
    structured_drs_test, claused_drs_test = extract_drs_features(test)
    frequency_counts = calculate_frequency(claused_drs)
    most_frequent = determine_highest(frequency_counts)
    gold_standard = get_gold_standard(claused_drs_test)
    only_lemmas = [[word_sense[0] for word_sense in sentence]
                   for sentence in gold_standard]
    prediction = disambiguate_sentences(
        only_lemmas, frequency_counts, most_frequent)
    gold_standard_reordered = match_disambiguation_order(
        gold_standard, frequency_counts)
    get_accuracy(gold_standard_reordered, prediction)


if __name__ == "__main__":
    main()
