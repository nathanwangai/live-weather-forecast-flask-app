from flask import Flask, render_template, request
import numpy as np
from joblib import load

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import uuid

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET': 
        return render_template('index.html', href='static/snowy_mountain.jpg', text_explanation = 'What will the weather be like today?')
    else:
        temp = request.form['temp']
        dew = request.form['dew']
        humidity = request.form['humidity']
        precip = request.form['precip']
        snowdepth = request.form['snowdepth']
        windspeed = request.form['windspeed']
        cloudcover = request.form['cloudcover']
        visibility = request.form['visibility']
        weather_data = np.array([temp, dew, humidity, precip, snowdepth, windspeed, cloudcover, visibility])
        weather_data = np.expand_dims(weather_data, axis=0)
        
        model_in = load('model.joblib')
        proba_list = (model_in.predict_proba(weather_data)*100).flatten().tolist()
        proba_df = pd.DataFrame({'class' : class_list,
                                 'probability' : proba_list})

        random_string = uuid.uuid4().hex
        path = 'static/' + random_string + '.jpg'
        make_bar_chart(path, proba_df)
        text = interpret_results(proba_df)

        return render_template('index.html', href=path, text_explanation = text)

# ------------------------------ non-flask code ---------------------------------

class_list = ['Clear', 'Overcast', 'Partially cloudy', 
              'Rain, Overcast', 'Rain, Partially cloudy', 
              'Snow, Overcast', 'Snow, Partially cloudy']

# probability bar chart of each class
def make_bar_chart(output_img_name, df):
    ax = sns.barplot(x='class', y='probability', data=df)
    plt.title('Prediction Probabilities', fontweight = 'heavy', fontsize=20)
    plt.ylabel('Probability (%)', fontweight = 'heavy', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=45, fontsize=10, horizontalalignment='right')
    plt.savefig(output_img_name, bbox_inches="tight", transparent=True)
    plt.show()

# explains the bar chart in plain English
def interpret_results(df):
    idx = df['probability'].idxmax()
    proba = df['probability'][idx]
    condition= df['class'][idx]
    
    if proba >= 90: 
        return 'It is almost definitely going to be "' + str(condition) + '"'
    elif proba >= 70:
        return 'There is a very good chance of being "' + str(condition) + '"'
    elif proba >= 50:
        return 'It will probably be "' + str(condition) + '"'

# convert comma separated values into numpy array
def string_to_array(input_string):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
        
    array = []
    elements = input_string.split(',')
        
    for i in elements:
        if is_float(i):
            array.append(i)
            
    return np.expand_dims(np.array(array), axis=0)

if __name__ == "__main__":
    app.run(debug=True)