import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads data from messages and categories files to merge them into a data frame
    
    Input:
        messages_filepath - String with the filepath of the messages information
        categories_filepath - String with the filepath of the categories information
    
    Output:
        df - Dataframe containing messages and categories
    """
    
    # Reading data from both sources
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merging the data
    df = messages.merge(categories, on = 'id')
    
    return df

def clean_data(df):
    """
    This function converts categories columns to binary variables and remove duplicates on the given dataframe
    
    Input: 
        df - Dataframe with data from categories
    
    Output:
        df - Dataframe after the cleaning process
    """
    # Create a dataframe of the categories columns
    categories = df['categories'].str.split(';', expand = True)
    
    # Extract a list of new column names for categories
    row = categories.loc[0]
    category_colnames = [r[:-2] for r in row]
    categories.columns = category_colnames
    
    # Converting categories to binary variables
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(float)
    
    # Drop the original categories column from df and adding the new columns
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, categories], axis = 1, sort = False)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    This function transform a dataframe into an sqlite database
    
    Input:
        df - Dataframe to be transformed
        database_filename - String with the name of the database
    """
    
    # Converting df to a sqlite database
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql('InsertTableName', engine, index=False) 


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