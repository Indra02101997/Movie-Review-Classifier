import nltk
import random
from nltk.corpus import movie_reviews
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
pos=open("positive.txt","r").read()
neg=open("negative.txt","r").read()

documents=[]
allowed_word_types=["J"]
all_words=[]

for r in pos.split('\n'):
    documents.append((r,"pos"))
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
        if(w[1][0] in allowed_word_types):
            all_words.append(w[0].lower())
for r in neg.split('\n'):
    documents.append((r,"neg"))
    words=word_tokenize(r)
    pos=nltk.pos_tag(words)
    for w in pos:
        if(w[1][0] in allowed_word_types):
            all_words.append(w[0].lower())

random.shuffle(documents)

save_documents=open("pickled_algos/documents.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()

#print(documents[0])              

all_words=nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

save_documents=open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features,save_documents)
save_documents.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
 
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
#featuresets = [(find_features(rev), category) for (rev, category) in documents]
featuresets=[]
for (rev,category) in documents:
    t=()
    t=(find_features(rev),category)
    featuresets.append(t)

random.shuffle(featuresets)

featureload=open("pickled_algos/featuresets5k.pickle","wb")
pickle.dump(featuresets,featureload)
featureload.close()

training_set=featuresets[:10000]
testing_set=featuresets[10000:]

classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Test set accuracy= ",(nltk.classify.accuracy(classifier,testing_set))*100.00)
#classifier.show_most_informative_features(15)
classifierf=open("pickled_algos/NaiveBayesClassifier5k.pickle","wb")
pickle.dump(classifier,classifierf)
classifierf.close()

MNBClassifier=SklearnClassifier(MultinomialNB())
MNBClassifier.train(training_set)
print("MNBClassifier accuracy= ",(nltk.classify.accuracy(MNBClassifier,testing_set))*100.00)

classifierf=open("pickled_algos/MNBClassifier5k.pickle","wb")
pickle.dump(MNBClassifier,classifierf)
classifierf.close()

BernoulliNBClassifier=SklearnClassifier(BernoulliNB())
BernoulliNBClassifier.train(training_set)
print("BernoulliNBClassifier accuracy= ",(nltk.classify.accuracy(BernoulliNBClassifier,testing_set))*100.00)

classifierf=open("pickled_algos/BernoulliNBClassifier5k.pickle","wb")
pickle.dump(BernoulliNBClassifier,classifierf)
classifierf.close()

LogisticRegressionClassifier=SklearnClassifier(LogisticRegression())
LogisticRegressionClassifier.train(training_set)
print("LogisticRegressionClassifier accuracy= ",(nltk.classify.accuracy(LogisticRegressionClassifier,testing_set))*100.00)

classifierf=open("pickled_algos/LogisticRegressionClassifier5k.pickle","wb")
pickle.dump(LogisticRegressionClassifier,classifierf)
classifierf.close()

SGD_Classifier=SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGDClassifierClassifier accuracy= ",(nltk.classify.accuracy(SGD_Classifier,testing_set))*100.00)

classifierf=open("pickled_algos/SGD_Classifier5k.pickle","wb")
pickle.dump(SGD_Classifier,classifierf)
classifierf.close()

SVCClassifier=SklearnClassifier(SVC())
SVCClassifier.train(training_set)
print("SVCClassifier accuracy= ",(nltk.classify.accuracy(SVCClassifier,testing_set))*100.00)

classifierf=open("pickled_algos/SVCClassifier5k.pickle","wb")
pickle.dump(SVCClassifier,classifierf)
classifierf.close()

LinearSVCClassifier=SklearnClassifier(LinearSVC())
LinearSVCClassifier.train(training_set)
print("LinearSVCClassifier accuracy= ",(nltk.classify.accuracy(LinearSVCClassifier,testing_set))*100.00)

classifierf=open("pickled_algos/LinearSVCClassifier5k.pickle","wb")
pickle.dump(LinearSVCClassifier,classifierf)
classifierf.close()

NuSVCClassifier=SklearnClassifier(NuSVC())
NuSVCClassifier.train(training_set)
print("NuSVCClassifier accuracy= ",(nltk.classify.accuracy(NuSVCClassifier,testing_set))*100.00)

classifierf=open("pickled_algos/NuSVCClassifier5k.pickle","wb")
pickle.dump(NuSVCClassifier,classifierf)
classifierf.close()

vote_classifier=VoteClassifier(classifier,MNBClassifier,BernoulliNBClassifier,LogisticRegressionClassifier,SGD_Classifier,LinearSVCClassifier,LinearSVCClassifier)
print("Voted Classifier accuracy = ",(nltk.classify.accuracy(vote_classifier,testing_set))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[0][0])," Confidence = ",(vote_classifier.confidence(testing_set[0][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[1][0])," Confidence = ",(vote_classifier.confidence(testing_set[1][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[2][0])," Confidence = ",(vote_classifier.confidence(testing_set[2][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[3][0])," Confidence = ",(vote_classifier.confidence(testing_set[3][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[4][0])," Confidence = ",(vote_classifier.confidence(testing_set[4][0]))*100)
print("Classifcation = ",vote_classifier.classify(testing_set[5][0])," Confidence = ",(vote_classifier.confidence(testing_set[5][0]))*100)






