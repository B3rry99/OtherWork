import os.path
from gensim import corpora
import gensim
import gensim.corpora as corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import csv
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import matplotlib.colors as mcolors
from datetime import datetime

# Loading in data and combining Title and Abstract
Train = pd.read_json("C:/Users/Jack/Desktop/Uni Work/ModellingAndOptimisation/trainjson.json")
Train["AbstractTitle"] = Train["TITLE"] + " - " + Train["ABSTRACT"]
AbTi = Train['AbstractTitle'].tolist()

# Removing symbols and numbers caused by encoding issues 
AbTi2 = []
for doc in AbTi:
    AbTi2.append(re.sub(r'[^\w]', ' ', doc))

AbTi3 = []
for doc in AbTi2:
    AbTi3.append(''.join([i for i in doc if not i.isdigit()]))
    
#writing as .txt file    
with open(r'C:/Users/Jack/Desktop/Uni Work/ModellingAndOptimisation/Data.txt', 'w',encoding="utf-8") as fp:
    for item in AbTi3:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

def load_data(path,file_name):
    """
    Input  : path and file_name
    Purpose: loading text file
    Output : list of paragraphs/documents and
             title(initial 100 words considred as title of document)
    """
    documents_list = []
    with open( os.path.join(path, file_name) ,"r", encoding="utf8") as fin:
        for line in fin.readlines():
            text = line.strip()
            documents_list.append(text)
    return documents_list

# Preprocessing data function
def preprocess_data(doc_set):

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    texts = []
    for i in doc_set:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    return texts

def prepare_corpus(doc_clean):
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary,doc_term_matrix

#create model
def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel

#calculate coherence values for different topic numbers
def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop=12, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

number_of_topics=6
words=10
# run model 
document_list=load_data("C:/Users/Jack/Desktop/Uni Work/ModellingAndOptimisation","Data.txt")
clean_text=preprocess_data(document_list)
startTime = datetime.now()
model=create_gensim_lsa_model(clean_text,number_of_topics,words)
print(datetime.now() - startTime)

# plot graph of different coherence values for varying topic numbers
def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

start,stop,step=6,13,1
plot_graph(clean_text,start,stop,step)
