# Importing essential libraries
import pandas as pd
import mlflow
from sklearn.metrics import classification_report, accuracy_score
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Loading the dataset

df = pd.read_csv('Spam SMS Collection', sep='\t', names=['label', 'message'])

mlflow.set_experiment("Spam Classifier Experiments")

# Cleaning the messages
corpus = []
ps = PorterStemmer()

for i in range(0, df.shape[0]):
    # Cleaning special character from the message
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.message[i])

    # Converting the entire message into lower case
    message = message.lower()

    # Tokenizing the review by words
    words = message.split()

    # Removing the stop words
    words = [word for word in words if word not in set(stopwords.words('english'))]

    # Stemming the words
    words = [ps.stem(word) for word in words]

    # Joining the stemmed words
    message = ' '.join(words)

    # Building a corpus of messages
    corpus.append(message)


cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

mlflow.sklearn.autolog()
with mlflow.start_run():
    classifier = MultinomialNB(alpha=0.3)
    classifier.fit(X_train, y_train)

    my_prediction = classifier.predict(X_test)
    print(my_prediction)
    print('Classification Report :\n', classification_report(y_test, my_prediction))
    print('Test Accuracy Score : ', accuracy_score(y_test, my_prediction))
    mlflow.sklearn.log_model(classifier, "my_model")

# filename = 'spam-sms-mnb-model.pkl'
# pickle.dump(classifier, open(filename, 'wb'))
