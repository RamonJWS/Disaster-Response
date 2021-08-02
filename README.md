# Disaster Response Pipeline Project

## Table of Contents
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [Components](#components)
 * [Instructions of How to Interact With Project](#instructions-of-how-to-interact-with-project)
 * [Licensing, Authors, Acknowledgements, etc.](#licensing-authors-acknowledgements-etc)
 
### Project Motivation
In this project, I used data provided by [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. I have created a machine learning pipeline to categorize real messages that were sent during disaster events so that the messages could be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### File Descriptions
app    

| - template    
| |- master.html # main page of web app    
| |- go.html # classification result page of web app    
|- run.py # Flask file that runs app    


data    

|- disaster_categories.csv # data to process    
|- disaster_messages.csv # data to process    
|- process_data.py # data cleaning pipeline    
|- InsertDatabaseName.db # database to save clean data to     


models   

|- train_classifier.py # machine learning pipeline     
|- classifier.pkl # saved model     


README.md    

### Components
There are three components I completed for this project. 

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline that:

 - Loads the messages and categories datasets
 - Merges the two datasets
 - Cleans the data
 - Stores it in a SQLite database
 
A jupyter notebook `ETL Pipeline Preparation` was used to do EDA to prepare the process_data.py python script. 
 
#### 2. ML Pipeline
A Python script, `train_classifier.py`, writes a machine learning pipeline that:

 - Loads data from the SQLite database
 - Splits the dataset into training and test sets
 - Builds a text processing and machine learning pipeline
 - Uses a custom Transformer Class to create new features
 - Trains and tunes a model using GridSearchCV
 - Outputs results on the test set
 - Exports the final model as a pickle file
 
A jupyter notebook `ML Pipeline Preparation` was used to do EDA to prepare the train_classifier.py python script. 

#### 3. Flask Web App
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. The outputs are shown below:

![app3](https://github.com/RamonJWS/Disaster-Response/blob/master/Images/front_pag.PNG)


![app1](https://github.com/RamonJWS/Disaster-Response/blob/master/Images/example1.PNG)


![app2](https://github.com/RamonJWS/Disaster-Response/blob/master/Images/example2.PNG)


### Instructions of How to Interact With Project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    
3. Open a new terminal, leaving "run.py" running, and input.  
    `env|grep WORK`

4. Go to Enter:
  `http://SPACEDID-3001.udacity-student-workspaces.com`

### Licensing, Authors, Acknowledgements, etc.
Thanks to Udacity for starter code for the web app. 
