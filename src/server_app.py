from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import requests
import json
from collections import Counter
from src.model import FraudClassifier, get_data
from src.database_operations import get_data_and_add_to_database
from src.database_operations import query_database
from pandas.io.json import json_normalize
import pandas as pd
import datetime
import pdb

#Initialize the app
app = Flask(__name__, static_url_path='/static')
#Load the pickled Model
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

# df = get_data(pd.read_json('../data/data.json'))
# # df['fraud_prob'] = model.predict_proba(df)
# # df['log_time'] = datetime.datetime(2000,1,1)

@app.route('/')
def index():
    

    return

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
