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
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report
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
    df = pd.read_sql_table('DisasterResponse', engine)
    # Features
    X = df['message']  
    # Target
    Y = df.iloc[:, 4:]
    return X, Y


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
        ('trasform', TfidfVectorizer(tokenizer=tokenize,sublinear_tf=True,
                                        stop_words = 'english',
                                        strip_accents='unicode',
                                        analyzer='word',
                                        token_pattern=r'\w{1,}',
                                        ngram_range=(1,4),
                                        dtype=np.float32,
                                        max_features = 10000)),
         
        ('classifier', MultiOutputClassifier(AdaBoostClassifier(
                                            base_estimator = DecisionTreeClassifier(class_weight='balanced'),
                                            learning_rate = 0.3,
                                            n_estimators = 100),
                                            random_state = 42))])

    # Set parameters for gird search
    parameters = {
        'classifier__estimator__learning_rate': [0.1, 0.2, 0.4],
        'classifier__estimator__n_estimators': [500, 1000, 2000]
    }

    # Set grid search
    grid_cv = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 5, scoring='f1_weighted', verbose = 2)

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