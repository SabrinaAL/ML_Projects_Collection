import sys
# import libraries
import pandas as pd
import numpy as np
import nltk
from pprint import pprint
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'universal_tagset'])
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from sklearn.metrics import confusion_matrix
from nltk.stem.snowball import SnowballStemmer 



def load_data(database_filepath):
    """
    Function: load data from sql database
    Args:
        database_filepath: database path
    Return:
        X (DataFrame) : Message features 
        Y (DataFrame) : target 
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)
    # Features
    X = df['message']  
    # Target
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    """
    Function: split text into words and return a list of tokenized words
    Args:
        text(str): the message
    Return:
        clean_tokens (list): a list of message words
    """
    
    # removing stop words and Stemming the remaining words in the message
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in text.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()


    tokens = word_tokenize(stemSentence)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Function: Build model.
    Returns:
          pipline: a sklearn estimator.
    """
    # Set pipeline
    pipeline = Pipeline([
        ('trasform', TfidfVectorizer(tokenizer=tokenize)),
         
        ('classifier', MultiOutputClassifier(AdaBoostClassifier(random_state = 42)))])

    # Set parameters for gird search
    parameters = {
        "classifier__estimator__learning_rate": [0.5, 1.0],
        "classifier__estimator__n_estimators": [20, 40]}

    # Set grid search
    grid_cv = GridSearchCV(estimator = pipeline, param_grid = parameters, verbose = 2, n_jobs = -1)

    return grid_cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Model evaluation
    Args:
        model: a sklearn estimator.
        X_test(array): messages.
        Y_test(array): categories for each message
        category_names(list): category names
    """

    y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print('Category {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        



def save_model(model, model_filepath):
    
    """
    Function: Save the model to a pickle file
    Args:
        model: the model
        model_filepath (str): pickle file path
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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