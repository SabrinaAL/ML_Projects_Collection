import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
      """
      Function:
      takes two csv files returns a merged dataFrame
      Args:
      messages_filepath (String): path to csv file that contains messages
      categories_filepath (String): path to csv file that contains categories
      Return:
      df (DataFrame): A dataframe of messages and categories
      """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    """Clean data.
    Args:
        df: pandas.DataFrame. dataFrame contains disaster messages and categories.
    Return:
        pandad.DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories
    categories = categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = list(categories.iloc[0])

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    # x = lambda row[i][0]: row[:-2] 
    category_colnames = list(map(lambda x: x[:-2], row))

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = list(map(lambda x: x[-1], list(categories[column])))

        # convert column from string to numeric
        categories[column] = list(map(lambda x: int(x), list(categories[column])))

        # Convert all value into binary (0 or 1) in related column
        
        if len(categories[column].value_counts()) > 2:
            for num in list(set(categories[column].values)):
                if num > 1:
                    categories[column].replace([num], [0], inplace=True)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Function:
    Save the Dataframe df in a sql database
    Args:
    df (DataFrame): messages and categories dataframe
    database_filename (str): database name
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()