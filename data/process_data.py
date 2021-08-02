import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(path1, path2):
    """
    Input: the path of the desired data, the first should be messages and the second categories.
    Output: both datasets merged together on id in as dataframes.
    """
    messages = pd.read_csv(path1)
    categories = pd.read_csv(path2)
    df = pd.merge(messages, categories, how="inner", on="id")
    
    return df

def clean_data(df):
    """
    Input: the categories dataframe
    Output: the categories dataframe with all columns being possible categories,
            and the rows being binary values 1=True 0=False.
    """
    regex = "[-01]"
    # create a list of column names
    columns = re.sub(regex, "", df.categories[0]).split(";")
    # create a dataframe of categories split on ";"
    categories = df.categories.str.split(pat=";", n=-1, expand=True)
    # iterates throught each row in the dataframe and keeps only the numeric values
    categories = categories.apply(lambda row: [x.split("-")[1] for x in row])
    # converts from string to int
    categories = categories.astype(int)
    # remove higher numbers
    categories = categories.apply(lambda row: [1 if x>=1 else 0 for x in row])
    # updates column names
    categories.columns = columns
    # removes the old categories from the original dataframe
    df.drop(labels="categories", inplace=True, axis=1)
    # merges the old dataframe with the newly created categories dataframe
    df = pd.concat([df,categories],axis=1)
    # removes dupilcate values
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filepath):
    """
    Input: the dataframe after removing duplicates, the name of the sql table you're creating.
    Output: Nothing.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql("DisasterResponse", engine, index=False, if_exists="replace")

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