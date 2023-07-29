from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        itching = int(request.form['itching'])
        continuous_sneezing = int(request.form['continuous_sneezing'])
        shivering = float(request.form['shivering'])
        joint_pain = int(request.form['joint_pain'])
        stomach_pain = int(request.form['stomach_pain'])
        vomiting = int(request.form['vomiting'])
        fatigue = float(request.form['fatigue'])
        weight_loss = float(request.form['weight_loss'])
        restlessness = float(request.form['restlessness'])
        Lethargy = float(request.form['Lethargy'])
        lack_of_concentration = float(request.form['lack_of_concentration'])


        values = np.array([[itching,continuous_sneezing,shivering,joint_pain,stomach_pain,vomiting,fatigue,weight_loss,restlessness,Lethargy,lack_of_concentration]])
        prediction = model.predict(values)
        prediction=prediction[0]
        return render_template('results.html', prediction=prediction)


if __name__ == "__main__":
    app.run(port=3000,debug=True)
