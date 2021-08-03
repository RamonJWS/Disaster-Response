import sys
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class NounCount(BaseEstimator, TransformerMixin):
    """
    Input: Inherits from the sklean.base class
    Creates: A transformer object that can be used to count the number of nouns in a corpus of documents
    """
    
    def count_nouns(self, text):
        list_of_noun_tags = ["NN", "NNP", "NNPS", "NNS"]
        noun_count = 0
        for word, tag in pos_tag(tokenize(text)):
            if tag in list_of_noun_tags:
                noun_count += 1
        return noun_count
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        text_transformed = pd.Series(X).apply(self.count_nouns)
        return pd.DataFrame(text_transformed)

def load_data(database_filepath):
    """
    Inputs: The database and table name.
    Outputs: The data split into train and test data for the model.
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df.message
    Y = df.drop(labels=["id","message","original","genre"],axis=1)
    
    return X,Y


def tokenize(text):
    """
    Inputs: A singel document of unedited text.
    Outputs: A single document of tokenized and standardized text.
    """
    # remove punctuation and tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    token = tokenizer.tokenize(text.lower())

    # remove stop words:
    stopwords_english = stopwords.words("english")
    tokens = [WordNetLemmatizer().lemmatize(word) for word in token if word not in stopwords_english]
    
    return tokens

def build_model():
    """
    Inputs: None.
    Outputs: Creates a data pipeline that transforms the data into a TF-IDF Vectorized form,
             while in parallel creates a new feature that counts the nouns. The pipeline 
             then merges these together and scales the data before using it with a
             RandomForeset Classifier. 
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
            ('count_noun', NounCount())
        ])),

        ('scale', StandardScaler(with_mean=False)),
        ('clf', RandomForestClassifier(n_jobs=-1))
    ])
    
    parameters = {"clf__n_estimators":[10,50,100]}
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    
    return model


def evaluate_model(model, X_test, Y_test):
    """
    Inputs: labelled data, the true value of Y data, and the predicted value of Y data.
    Returns: the classification report for all the labels.
    """
    
    y_pred = model.predict(X_test)
        
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    Input: The trained model and the name given to the model.
    Creates: Saves the model to a pickle file to be used in the Flask app.
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()