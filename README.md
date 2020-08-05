# news-qa
** Before running any of the files, please run "pip install -r requirements.txt" **

Please review the files in the following order:

1. EDA.ipynb - Exploratory Data Analysis on original data and formatting/initial cleaning
2. Data Cleaning.ipynb - Cleaning the data
3. Baseline Model.ipynb - A baseline model with cosine similarity
4. newsqa.py - Functions and classes for pre-processing the data and model training
5. Advanced Modelling.ipynb - Data preparation and fine-tuining BERT and DistilBERT models
~.  utils.py - Just some utility functions commonly used in above files
6. main.py - FastAPI deployment
7. front-end/newsqa.html - The front-end UI code

To run the code present in files 1 to 5 on a sample of the data, use "Demo Code.ipynb"

To test the API, run "uvicorn main:app --reload" and visit "http://127.0.0.1:8000/docs" or whatever port the app is running on your local machine. Alternatively, you can also visit "https://fastapi-newsqa.wl.r.appspot.com/docs" where the API is deployed. You need to have a trained model with path "data/bert_model.pt".

To view the front end, just open the file "front-end/newsqa.html" on a browser with a working internet connection. Alternatively, you can also visit "smitkiri.github.io/newsqa" where the front end is deployed.