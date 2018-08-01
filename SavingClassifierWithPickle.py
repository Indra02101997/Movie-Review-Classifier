import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents=[]
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        t=()
        t=(list(movie_reviews.words(fileid)),category)
        documents.append(t)

random.shuffle(documents)

#print(documents[0])
all_words=[]

for w in movie_reviews.words():
                         all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
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

training_set=featuresets[:1900]
testing_set=featuresets[1900:]

#classifier=nltk.NaiveBayesClassifier.train(training_set)
classifier_f=open('naivebayesclassifier','rb')
classifier=pickle.load(classifier_f)
classifier_f.close()

print("Test set accuracy= ",(nltk.classify.accuracy(classifier,testing_set))*100.00)
classifier.show_most_informative_features(15)

#saving the classifier
#save_classifier=open('naivebayesclassifier','wb')
#pickle.dump(classifier,save_classifier)
#save_classifier.close()











                         
                         
