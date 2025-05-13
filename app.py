from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('PlacementPredictor.pkl')  # Make sure your model is trained and saved as model.pkl

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['CGPA']),
            int(request.form['Internships']),
            int(request.form['Projects']),
            int(request.form['Workshops']),
            int(request.form['AptitudeTestScore']),
            float(request.form['SoftSkillsRating']),
            int(request.form['ExtracurricularActivities']),
            int(request.form['PlacementTraining']),
            int(request.form['SSC_Marks']),
            int(request.form['HSC_Marks']),
        ]
        prediction = model.predict([data])
        result = "Placed" if prediction[0] == 1 else "Not Placed"
        return render_template('form.html', prediction=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
