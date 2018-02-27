"""
Created on Sat Nov 25 22:41:19 2017

@author: Sakshi, Shobhit
"""
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


print("\n\n___start___")
"""
Reading tweets from test file
"""
tweets = pd.read_csv('sample_test.csv', encoding = "ISO-8859-1", names=['id','Anootated tweet'],  dtype=str)

"""
Test Data Preprocessing
"""
print('Data preprocessing started...')
#RegEx
re.compile('<.*?>')
TAG_RE = re.compile(r'<[^>]+>')

#Stopwords
stop_words = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}
stop_words_pat = '|'.join(['\\b' + stop +  '\\b' for stop in stop_words])

#Stemming
ps = PorterStemmer()
def stemming(x):
    final_tweet = ""
    wordlist = nltk.word_tokenize(x)
    for word in wordlist:
        word = ps.stem(word)
        final_tweet += ' '+ word
    return final_tweet

#Cleaning
tweets['Anootated tweet'] = tweets['Anootated tweet'].apply(lambda x: str(x).lower())
tweets['Anootated tweet'] = tweets['Anootated tweet'].apply((lambda x: re.sub(r"http\S+", "", str(x))))
tweets['Anootated tweet'] = tweets['Anootated tweet'].apply((lambda x: TAG_RE.sub('', str(x))))
tweets['Anootated tweet'] = tweets['Anootated tweet'].apply((lambda x: re.sub('[^a-zA-Z\s]','',str(x))))
tweets['Anootated tweet'] = tweets['Anootated tweet'].apply((lambda x: re.sub(stop_words_pat,'',str(x))))
tweets['Anootated tweet'] = tweets['Anootated tweet'].apply((lambda x: stemming(x)))

"""
Reading tweets from training file
"""
f = pd.read_csv('Obama.csv', sep=',', names=['Anootated tweet', 'Class'], dtype=str)
f.drop(f.index[0])


"""
Training Data Preprocessing
"""
f['Anootated tweet'] = f['Anootated tweet'].apply(lambda x: str(x).lower())
f['Anootated tweet'] = f['Anootated tweet'].apply((lambda x: re.sub(r"http\S+", "", str(x))))
f['Anootated tweet'] = f['Anootated tweet'].apply((lambda x: TAG_RE.sub('', str(x))))
f['Anootated tweet'] = f['Anootated tweet'].apply((lambda x: re.sub('[^a-zA-Z\s]','',str(x))))
f['Anootated tweet'] = f['Anootated tweet'].apply((lambda x: re.sub(stop_words_pat,'',str(x))))
f['Anootated tweet'] = f['Anootated tweet'].apply((lambda x: stemming(x)))

print('Data preprocessing ended...')

def split_into_lemmas(tweet):
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 5), token_pattern=r'\b\w+\b', min_df=1)
    analyze = ngram_vectorizer.build_analyzer()
    return analyze(tweet)

print('transformation started...')
bow_transformer = CountVectorizer(analyzer = split_into_lemmas, stop_words='english', strip_accents='ascii').fit(f['Anootated tweet'])
text_bow = bow_transformer.transform(f['Anootated tweet'])
tfidf_transformer = TfidfTransformer().fit(text_bow)
tfidf = tfidf_transformer.transform(text_bow)
text_tfidf = tfidf_transformer.transform(text_bow)
print('transformation ended...')


"""
Classifier
"""
print('Training started...')
classifier_nb = LogisticRegression().fit(text_tfidf, f['Class'])
print('Classes found: ',classifier_nb.classes_)
sentiments = pd.DataFrame(columns = ['class'])
print('Training ended...')
"""
Prediction based on classifier
"""
print('Predicting for test data...')
i = 0
for _, tweet in tweets.iterrows():
    i += 1
    try:
        bow_tweet = bow_transformer.transform(tweet)
        tfidf_tweet = tfidf_transformer.transform(bow_tweet)
        sentiments.loc[i-1, 'id'] = tweet.values[0]
        sentiments.loc[i-1, 'class'] = classifier_nb.predict(tfidf_tweet)[0]
        round(classifier_nb.predict_proba(tfidf_tweet)[0][1], 2)*10
    except Exception as e:
        print('xxxx error xxxx\n',sys.exc_info(),'\n')


'''
Generating Output File
'''
sentiments.set_index('id', inplace=True)
header = ["class"]
sentiments.to_csv('output.txt', encoding ='utf-8', columns = header, header=None)

print('Output File Generated\n___end___\n\n')

predicted_classes = sentiments.loc[: , "class"]

