from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetError
import os

import re
# pmb_path = 'PMB/parsing/layer_data/4.0.0/en/gold/dev.conll'
# pmb_path = 'PMB/parsing/layer_data/4.0.0/en/gold/eval.conll'
# pmb_path = 'PMB/parsing/layer_data/4.0.0/en/gold/test.conll'
pmb_path = 'PMB/parsing/layer_data/4.0.0/en/gold/train.conll'
consec_path = 'consec/data'

with open(pmb_path) as file:
        data = file.read()
        data = data.split('\n\n')
        data.pop(-1)    
        
fname = pmb_path.split('/')[-1]
ftype = fname.split('.')[0]
consec_path = os.path.join(consec_path, ftype +'.xml')
# print(consec_path)
consec_file = open(consec_path, "w")
consec_file.write("<contextfile concordance=\"pmb\">\n")
consec_file.write("<contextfile concordance=\"" + fname + "\" paras=\"yes\">\n")
lemmatizer = WordNetLemmatizer()
for doc_no, doc in enumerate(data,1):
    if doc_no == 1:
        doc = '\n'.join(doc.split('\n')[2:])
    if len(word_tokenize(' '.join(doc.split('\n')[3].split(' ')[4:]))) != len(doc.split('\n'))-4:
        continue
    flag = 0
    for line_no, line in enumerate(doc.split('\n'), 1):
        if line_no == 1:
            if "newdoc" in line:
                m = re.match("# newdoc id = p([0-9]*)/d([0-9]*)", line)
                sent = m.group(1)
                para = m.group(2)
                consec_file.write("<p pnum=\"" + para+ "\">\n")
                consec_file.write("<s snum=\"" + sent+ "\">\n")
            else:
                # print('broke')
                flag = 1
                break
        if line_no == 4:
            sentence = ' '.join(line.split(' ')[4:])
            tokens =  word_tokenize(sentence)
            tokenized = pos_tag(tokens)
        if line_no > 4:
            features = {}
            items = line.split('\t')
            word = items[0]
            lemma = items[2]
            wnsense = items[5]
            # print(sentence)
            # print(tokenized)
            pos = tokenized[line_no-5][1]
            if wnsense != 'O':
                wnsn = str(int(wnsense.split('.')[2]))
                try:
                    lexsn = str(wn.synset(wnsense).lemmas()[0].key().split('%')[1])
                    consec_file.write("<wf cmd=\"done\" pos=\"" + pos +"\" lemma=\"" +lemma+"\" wsn=\"" +wnsn+"\" lexsn=\""+lexsn+"\">"+word+"</wf>\n")
                except WordNetError:
                    try:
                        lemma = lemmatizer.lemmatize(lemma)
                        lexsn = str(wn.synset(lemma + "." + ".".join(wnsense.split('.')[1:])).lemmas()[0].key().split('%')[1])
                        consec_file.write("<wf cmd=\"done\" pos=\"" + pos +"\" lemma=\"" +lemma+"\" wsn=\"" +wnsn+"\" lexsn=\""+lexsn+"\">"+word+"</wf>\n")
                    except WordNetError:
                         print(lemma)
                         consec_file.write("<wf cmd=\"done\" pos=\"" + pos +"\" lemma=\"" +lemma+"\" wsn=\"" +wnsn+"\">"+word+"</wf>\n")
            else:
                if pos =='.':
                    consec_file.write("<punc>" + word + "</punc>\n")
                else:
                    consec_file.write("<wf cmd=\"ignore\" pos=\"" + pos +"\">"+word+"</wf>\n")
    if flag == 0:
        consec_file.write("</s>\n")
        consec_file.write("</p>\n")
consec_file.close()     
        