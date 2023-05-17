import numpy as np
import re
import string
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
import nltk
from nltk.corpus import stopwords
from itertools import chain
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import conll2000
import pickle
import nltk
import nltk.data
from readability import Readability

chunker_instance = None
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tagger = nltk.data.load("pos_tagger/treebank_brill_aubt.pickle")

grammar = r"""
  NP: 
      {<DT|WDT|PP\$|PRP\$>?<\#|CD>*(<JJ|JJS|JJR><VBG|VBN>?)*(<NN|NNS|NNP|NNPS>(<''><POS>)?)+}
      {<DT|WDT|PP\$|PRP\$><JJ|JJS|JJR>*<CD>}
      {<DT|WDT|PP\$|PRP\$>?<CD>?(<JJ|JJS|JJR><VBG>?)}
      {<DT>?<PRP|PRP\$>}
      {<WP|WP\$>}
      {<DT|WDT>}
      {<JJR>}
      {<EX>}
      {<CD>+}
  VP: {<VBZ><VBG>}
      {(<MD|TO|RB.*|VB|VBD|VBN|VBP|VBZ>)+}
      

"""

def get_nltk_pos_tag_based_chunker():
    global chunker_instance
    if chunker_instance is not None:
        return chunker_instance
    chunker_instance = nltk.RegexpParser(grammar)
    return chunker_instance

    

def chunk_to_str(chunk):
    if type(chunk) is nltk.tree.Tree:
        return chunk.label()
    else:
        return chunk[1]

def extract_subtree_expansions(t, res):
    if type(t) is nltk.tree.Tree:
        expansion = t.label() + "[" + " ".join([chunk_to_str(child) for child in t]) + "]"
        res.append(expansion)
        for child in t:
            extract_subtree_expansions(child, res)
            
def nltk_pos_tag_chunk(pos_tags):
    chunker = get_nltk_pos_tag_based_chunker()
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions


def prepare_entry(text):
    tokens = []
    # Workaround because there re some docuemtns that are repitions of the same word which causes the regex chunker to hang
    prev_token = ''
    for t in tokenizer.tokenize(text):
        if t != prev_token:
            tokens.append(t)
    tagger_output = tagger.tag(tokens)
    pos_tags = [t[1] for t in tagger_output]
    pos_chunks, subtree_expansions = nltk_pos_tag_chunk(tagger_output)
    entry = {
        'preprocessed': text,
        'pos_tags': pos_tags,
        'pos_tag_chunks': pos_chunks,
        'pos_tag_chunk_subtrees': subtree_expansions,
        'tokens': tokens
    }
    return entry



def word_count(entry):
    return len(entry['tokens'])

def avg_chars_per_word(entry):
    r = np.mean([len(t) for t in entry['tokens']])
    return r

def distr_chars_per_word(entry, max_chars=10):
    counts = [0] * max_chars
    for t in entry['tokens']:
        l = len(t)
        if l <= max_chars:
            counts[l - 1] += 1
    r = [c/len(entry['tokens']) for c in counts]
#     fnames = ['distr_chars_per_word_' + str(i + 1)  for i in range(max_chars)]
    return r
    
def character_count(entry):
    r = len(re.sub('\s+', '', entry['preprocessed']))
    return r


#https://github.com/ashenoy95/writeprints-static/blob/master/whiteprints-static.py
def hapax_legomena(entry):
    freq = nltk.FreqDist(word for word in entry['tokens'])
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    if len(dis) == 0 or len(entry['tokens']) == 0:
        return 0
    return (len(hapax) / len(dis)) / len(entry['tokens'])

def n_of_pronouns(entry):
    # Gab: aggiunto da noi
    return entry['pos_tags'].count('PRP') + entry['pos_tags'].count('PRP$')

def gunning_fog_index(entry):
    # Gab: aggiunto da noi
    r = Readability(entry['preprocessed'])
    gf = r.gunning_fog()
    return gf.score


def pass_fn(x):
    return x

class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, key, analyzer, n=1, vocab=None):
        self.key = key
        if self.key == 'pos_tags' or self.key == 'tokens' or self.key == 'pos_tag_chunks' or self.key == 'pos_tag_chunk_subtrees':
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm='l1', ngram_range=(1, n))
        else:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1, vocabulary=vocab, norm='l1', ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry[self.key] for entry in x], y)
        return self

    def transform(self, x):
        return self.vectorizer.transform([entry[self.key] for entry in x])
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    
class CustomFreqTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, analyzer, n=1, vocab=None):
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn, vocabulary=vocab, norm=None, ngram_range=(1, n))

    def fit(self, x, y=None):
        self.vectorizer.fit([entry['tokens'] for entry in x], y)
        return self

    def transform(self, x):
        d = np.array([1 + len(entry['tokens']) for entry in x])[:, None]
        return self.vectorizer.transform([entry['tokens'] for entry in x]) / d
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()
    
    
class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames
        
    def fit(self, x, y=None):
        return self;
    
    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        else:
            return xx
    
    def get_feature_names(self):
        if self.fnames is None:
            return ['']
        else:
            return self.fnames
        
        
def get_writeprints_transformer():
    char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', n=6)
    word_distr = CustomTfIdfTransformer('preprocessed', 'word', n=3)
    pos_tag_distr = CustomTfIdfTransformer('pos_tags', 'word', n=3)
    pos_tag_chunks_distr = CustomTfIdfTransformer('pos_tag_chunks', 'word', n=3)
    pos_tag_chunks_subtree_distr = CustomTfIdfTransformer('pos_tag_chunk_subtrees', 'word', n=1)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{¦}~'
    special_char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', vocab=punctuation)
    freq_func_words = CustomFreqTransformer('word', vocab=stopwords.words('english'))

    transformer = FeatureUnion([
        ('char_distr', char_distr),
        ('word_distr', word_distr),
        ('pos_tag_distr', pos_tag_distr),
        ('pos_tag_chunks_distr', pos_tag_chunks_distr),
        ('pos_tag_chunks_subtree_distr', pos_tag_chunks_subtree_distr),
        ('special_char_distr', special_char_distr),
        ('freq_func_words', freq_func_words),
        ('hapax_legomena', CustomFuncTransformer(hapax_legomena)),
        ('character_count', CustomFuncTransformer(character_count)),
        ('distr_chars_per_word', CustomFuncTransformer(distr_chars_per_word, fnames=[str(i) for i in range(10)])),
        ('avg_chars_per_word', CustomFuncTransformer(avg_chars_per_word)),
        ('word_count', CustomFuncTransformer(word_count)),
        ('readability', CustomFuncTransformer(gunning_fog_index) ),
        ('n_of_pronouns', CustomFuncTransformer(n_of_pronouns) )
    ], n_jobs=-1)
    
    return transformer

