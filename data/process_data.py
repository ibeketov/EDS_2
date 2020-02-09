import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Lading data from csv files
    
    Arguments:
        messages_filepath {[string]} -- path to the messages dataset
        categories_filepath {string} -- path to the categories dataset
    
    Returns:
        Merged dataframe of both datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    """Cleans the dataset
    1. Split categories into separate category columns
    2. Convert category values to just numbers 0 or 1
    3. 
    Arguments:
        df {DataFrame} -- dateset
    
    Returns:
        DataFrame -- resulted dataset
    """    
    categories = df['categories'].str.split(';', expand=True)
    # use this row to extract a list of new column names for categories.
    row = categories.loc[1]
    category_colnames = row.apply(lambda i: i[:-2])
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1::]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    df = df.drop(['categories'], axis=1)
    df= pd.concat([df, categories], axis=1)
    
    # drop duplicate
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Save to db, if exist will be replased
    
    Arguments:
        df -- dataset
        database_filename {Str} -- db file name
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseData', engine, index=False, if_exists='replace')

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