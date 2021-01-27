# Disaster Response Pipeline Project

## Installation

Code was developed using Python 3.8.3.

libraries needed:

- pandas
- numpy
- matplotlib
- seaborn
- nltk
- scikit-learn
- sqlalchemy

## Project Description
In this project, I analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that the messages are sent to an appropriate disaster relief agency. The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

Here is a visualization of frequent words that appear in each category:

![alt text](https://github.com/SabrinaAL/Data-Science-Nanodegree/blob/master/disaster_response/important_words.png)

Screenshot of the app home page:

![alt text](https://github.com/SabrinaAL/Data-Science-Nanodegree/blob/master/disaster_response/Training%20DataSet.png)

Screenshot of the predicted categories for message "We need water":

![alt text](https://github.com/SabrinaAL/Data-Science-Nanodegree/blob/master/disaster_response/ExampleOutput.png)


## File Descriptions

- Data Analysis .ipynb: Useful mix of scripts for analysis and vizualization of the disaster messages
- workspace/ETL Pipeline Preparation.ipynb: workspace to create python script for workspace/data/process_data.py
- workspace/ML Pipeline Preparation.ipynb: workspace to create python script for workspace/model/train_classifier.py
- workspace/data/process_data.py: A data cleaning pipeline that cleans data and stores in database
- workspace/model/train_classifier.py: A machine learning pipeline that trains classifier and saves

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
