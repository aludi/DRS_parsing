# DRS_parsing
This file contains information on how to run, experiment with, and evaluate the different baselines and models for the final project for Computational Semantics. For questions or concerns please reach out to:

Robert van Timmeren - r.j.van.timmeren@student.rug.nl

<h2>About</h2>
We conduct Word Sense Disambiguation (WSD) on the English gold data of the Parallel Meaning Bank (PMB). Our programs are divided into; two baselines and three classification models.
The first baseline predicts word senses based on the first entry on WordNet, in line with the correct part of speech tag. 
Our second baseline calculates frequencies of correctly annotated WordNet senses on the test and dev set. It then annotates a word sense based on the most frequently occurring WordNet sense in the test and dev set, if it occurred at all. Otherwise, it simply takes the first WordNet entry again.Our first two models use naive similarity with one model using a similarity approach with Wu-Palmer similarity (wup-similarity). Our second model compares distance between word-vectors.
Our third and last model uses a hybrid approach by relying on frequency for monosemous words and distance calculating with the use of Wu-Palmer similarity again.

<h2>Installation of dependencies</h2>
The required dependencies can be installed with:

```
pip install -r requirements.txt
```

<h2>Files explanation</h2>

File  | Explanation
------------- | -------------
Baseline/baseline.py  | Contains the two baseline models.
Models/improved_model.py  | Hybrid approach model.
Models/model.py  |  Wu-Palmer similarity model.
Models/similarity.py  |  Word-vector similarity model.
pmb_to_semcor_format.py  |  Restructures the PMB file to semcor format.
PMB/  |  The PMB data containing for our programs.

<h2>Notable results</h2>

Model  | File  |  Accuracy (%)
------------- | ------------- | -------------
1st sense baseline  |  Baseline/baseline.py  |  33.98%
Frequency baseline  |  Baseline/baseline.py  |  86.62%
WUP similarity model  |  Models/model.py  |  69.78%
Word vector similarity model  |  Models/similarity.py  |  70.17%
Hybrid model  |  Models/improved_model.py  |  83.14%
