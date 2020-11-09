# Disaster Response Pipeline Project

Predicting necessary services from messages.

## Index

1. [Requirements and Instructions](#requirements)
3. [Project](#project)
4. [Files](#files)
5. [Acknowledgements](#acknowledgements)

<a name="requirements"></a>
### Requirements
The libraries used in this project are:

jplotly 4.8.2, pandas 1.0.1, nltk 3.4.5, flask 1.1.1, sklearn 0.22.1, sqlachemy 1.3.13

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="projects"></a>
### Project

This project builds am ETL pipeline to transform raw tweets and messages into inputs for a machine learning model that classifies them into categories. This is important for a quick response from the organization in charge of the disaster specific task such as medical supplies.

For this project we used a pipeline with two steps:

1. Text wrangling. In this step, we used countvectorizer and TfidfTransformer functions to obtain the lemmas of the message.
2. Machine learning model. After having the lemmas as inputs, we trained a Random Forest to output the correct classification of the message.

<a name="files"></a>
### Files

This repository contains the following folders and files:

* [app](https://github.com/MauricioTrejo/DisasterResponse/tree/master/app). Contains the templates for the web app (go.html and master.html), as well as, run.py, the file needed to launch the app.

* [data](https://github.com/MauricioTrejo/DisasterResponse/tree/master/data). Contains 2 csv files with messages and categories, this files are used in a ETL pipeline saved in process_data.py, which after processing both csv files, return a sqlite database.

* [models](https://github.com/MauricioTrejo/DisasterResponse/tree/master/models). Contains train_classifier.py, where the Random Forest to make the classification of messages lies.

<a name="acknowledgements"></a>
### Acknowledgements

We want to thank [Figure Eight](https://appen.com/?ref=Welcome.AI) for the data used in this project.
