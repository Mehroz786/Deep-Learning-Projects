This folder contains a Flask web application that predicts the category of the articles.

Folders
models
This folder contains the deep learning model implemented using Keras and saved as a pickle file (Reuters.pkl). The model is loaded in the Flask application to make predictions.

templates
This folder contains the HTML templates for the web application. The index.html file is the main template that includes a form for users to input their information and submit it to the server for prediction. And result will be show after the click of the predict btn. 

Files
app.py
This is the main file of the Flask application that handles the web requests and makes predictions using the deep learning model. It loads the model from the models folder and renders the HTML templates from the templates folder.