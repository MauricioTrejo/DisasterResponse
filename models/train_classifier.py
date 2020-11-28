import sys
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt', 'stopwords', 'wordnet'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split, GridSearchCV

import pickle


def load_data(database_filepath):
    """
    This function loads data from an sql database and transform it into a pandas dataframe
    
    Input:
        database_filepath - String with the name of the database
        
    Output:
        X - DataFrame with the messages to be classified
        Y - Dataframe with the labels
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.drop(labels = ['id', 'original', 'genre', 'message'], axis = 1)
    
    return X, Y, Y.columns


def tokenize(text):
    """
    Function that tokenize and lemmatize the given text
    
    Input:
        String with the text to be transformed
    
    Output:
        String with the lemmas of the string given as input
    """
    
    # Loading the stop words and Lemmatizer
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # Transform to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the resulting text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    This functions makes the model to classify messages, it has a countvectorizer, a tfidftransformer
    for the data processing step and a random forest for the model step
    
    Output:
        Pipeline with the steps of the pipeline 
    """
    
    pipeline = Pipeline(steps = [
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('model', MultiOutputClassifier(RandomForestClassifier(random_state = 17)))
    ])
    
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    This functions prints the classification report for each category name
    
    Input:
        model - The model to be tested
        X_test - Dataframe with the input variables to be predicted and tested for model performance
        Y_test - Dataframe containing the true values of the classification
        category_names - Array with the name of each category
    """
    
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)
    
    for i in range(y_test.shape[1]):
        print('Classification report for label: ' + category_names[i])
        print(classification_report(y_test.iloc[:,i], y_pred.iloc[:,i], labels = [0, 1]))
    

def save_model(model, model_filepath):
    """
    This function saves the model as a pickle file
    
    Input:
        model - The model to save
        model_filepath - String with the filepath in which the model will be saved
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