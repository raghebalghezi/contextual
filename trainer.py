# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context
import pickle
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support

data_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)

y_train, y_test = data_train.target, data_test.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=33809)

X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

print("The shape of X_train is {}".format(X_train.shape))
print("The shape of y_train is {}".format(y_train.shape))

# classifier = SGDClassifier(n_jobs=-1, alpha=.0001, max_iter=50, penalty='l2')
classifier = PassiveAggressiveClassifier()

classifier.fit(X_train, y_train)

y_hat = classifier.predict(X_test)


topics = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 
'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 
'talk.politics.misc', 'talk.religion.misc']

# report = classification_report(y_test, y_hat, labels=topics, zero_division='warn')

report = precision_recall_fscore_support(y_test, y_hat, average='weighted', warn_for=tuple())

print(report)
# print(twenty_train.target_names)

# pickle.dump(classifier, open("classifier.pkl", 'wb'))

# pickle.dump(vectorizer, open("vectorizer.pkl", 'wb'))

