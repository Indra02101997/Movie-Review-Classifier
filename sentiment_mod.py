import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import pickle

class VoteClassifier(ClassifierI):
    def __init__ (self,*classifier):
        self._classifiers=classifier
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice=votes.count(mode(votes))
        conf=(choice/len(votes))
        return conf


save_documents=open("pickled_algos/documents.pickle","rb")
documents=pickle.load(save_documents)
save_documents.close()

save_documents=open("pickled_algos/word_features.pickle","rb")
word_features=pickle.load(save_documents)
save_documents.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
 
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
#featuresets = [(find_features(rev), category) for (rev, category) in documents]


featureload=open("pickled_algos/featuresets5k.pickle","rb")
featuresets=pickle.load(featureload)
featureload.close()

random.shuffle(featuresets)

classifierf=open("pickled_algos/NaiveBayesClassifier5k.pickle","rb")
classifier=pickle.load(classifierf)
classifierf.close()


classifierf=open("pickled_algos/MNBClassifier5k.pickle","rb")
MNBClassifier=pickle.load(classifierf)
classifierf.close()

classifierf=open("pickled_algos/BernoulliNBClassifier5k.pickle","rb")
BernoulliNBClassifier=pickle.load(classifierf)
classifierf.close()

classifierf=open("pickled_algos/LogisticRegressionClassifier5k.pickle","rb")
LogisticRegressionClassifier=pickle.load(classifierf)
classifierf.close()

classifierf=open("pickled_algos/SGD_Classifier5k.pickle","rb")
SGD_Classifier=pickle.load(classifierf)
classifierf.close()

classifierf=open("pickled_algos/SVCClassifier5k.pickle","rb")
SVCClassifier=pickle.load(classifierf)
classifierf.close()

classifierf=open("pickled_algos/LinearSVCClassifier5k.pickle","rb")
LinearSVCClassifier=pickle.load(classifierf)
classifierf.close()

classifierf=open("pickled_algos/NuSVCClassifier5k.pickle","rb")
NuSVCClassifier=pickle.load(classifierf)
classifierf.close()

vote_classifier=VoteClassifier(classifier,MNBClassifier,BernoulliNBClassifier,LogisticRegressionClassifier,SGD_Classifier,LinearSVCClassifier,LinearSVCClassifier)

def sentiment(text):
    feats=find_features(text)
    return vote_classifier.classify(feats),vote_classifier.confidence(feats)





