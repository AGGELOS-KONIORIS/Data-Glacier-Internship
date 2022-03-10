#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods = ['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])
def predict():
    if request.method == 'POST':
        Ratings = int(request.form['Ratings'])
        Age = int(request.form['Age'])
        Minutes_Played = float(request.form['Minutes_Played'])
        Fieldgoal_Percentage = float(request.form['Fieldgoal_Percentage'])
        Threepoint_Percentage = float(request.form['Threepoint_Percentage'])
        Twopoint_Percentage = float(request.form['Twopoint_Percentage'])
        Freethrow_Percentage = float(request.form['Freethrow_Percentage'])
        Total_Rebounds = float(request.form['Total_Rebounds'])
        Asists = float(request.form['Asists'])
        Steals = float(request.form['Steals'])
        Blocks = float(request.form['Blocks'])
        Turnovers = float(request.form['Turnovers'])
        Personal_Fouls = float(request.form['Personal_Fouls'])
        Points = float(request.form['Points'])
        
        prediction = model.predict([[Ratings, Age, Minutes_Played, Fieldgoal_Percentage, Threepoint_Percentage, Twopoint_Percentage, Freethrow_Percentage, Total_Rebounds, Asists, Steals, Blocks, Turnovers, Personal_Fouls, Points]])
        output = round(prediction[0], 2)
        if output < 0:
            return render_template('index.html', prediction_texts = "Sorry you might do something wrong")
        else:
            return render_template('index.html', prediction_text = "The player's salary will be around:{}$".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug = True)

