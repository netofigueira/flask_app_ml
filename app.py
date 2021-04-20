from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import pandas as pd
import requests
import sklearn
from features import  features

#requests.get('https://api.mercadolibre.com/items/{}'.format(item_id))
app = Flask(__name__)


######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_model/poisson_model_ml.pkl'), 'rb'))

class ItemForm(Form):
    item_to_predict = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = ItemForm(request.form)
    return render_template('form_app.html', form=form)


@app.route('/predict', methods=['POST'])
def predict():
    form = ItemForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['item_to_predict']
        #r =requests.get('https://api.mercadolibre.com/items/{}'.format(name)).json()
        item_df =items_preprocess(name, features)

        pred = predict_qtt(item_df)
        actual_value = item_df['sold_quantity'][0]
        return render_template('predict.html', name=pred[0], value=actual_value)
    
    return render_template('form_app.html', form=form)


def features_build(df):
    df['discount_value'] = df['original_price'] - df['price']
    df['discount_pct'] = df['discount_value']/df['original_price']
    df['sold_per_available'] = df['sold_quantity']/df['available_quantity']
    df['has_discount'] = df.discount_value.apply(lambda val: 'no' if str(val) == 'nan' else 'yes')
    df['discount_value'].fillna(value=0, inplace=True)
    df['discount_pct'].fillna(value=0, inplace=True)
    return df



def items_preprocess(item_id, features):

    r =requests.get('https://api.mercadolibre.com/items/{}'.format(item_id)).json()
    item_df = pd.json_normalize(r)
    item_df['available_quantity'][0] = item_df['initial_quantity'][0]
    features_build(item_df)
    return item_df[features]

def predict_qtt(item):
    X = item
    y = clf.predict(X)
    return np.round(y * X['available_quantity'][0])


if __name__ == '__main__':
    app.run(debug=True)