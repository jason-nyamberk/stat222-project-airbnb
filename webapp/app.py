import flask
import pickle

# Use pickle to load in the pre-trained model.
with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        location = flask.request.form['location']
        bedrooms = flask.request.form['bedrooms']
        
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       columns=['location', 'bedrooms'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Location':location,
                                                     'Bedrooms':bedrooms},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()