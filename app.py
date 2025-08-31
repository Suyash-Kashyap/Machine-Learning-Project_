import joblib
from flask import Flask, request, render_template_string
import os
from flask import Flask, request, render_template_string

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
OCC_PATH   = os.path.join(MODEL_DIR, 'occupation_label_encoder.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')  # optional if you want

# Load model and encoders
model = joblib.load(MODEL_PATH)
le_occ = joblib.load(OCC_PATH)


app = Flask(__name__)

home_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Sleep Disorder Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #182c25; color: #f1f1f1; }
        .container { width: 600px; margin: 40px auto; background: #213d32; padding: 30px; border-radius: 10px; }
        h1 { text-align: center; color: #ffcd38; }
        label { display: block; margin-top: 10px; }
        input, select { width: 100%; padding: 8px; margin-top: 5px; }
        button { margin-top: 20px; width: 100%; padding: 12px; background: #ffcd38; border: none; cursor: pointer; }
        button:hover { background: #ffe18a; }
    </style>
</head>
<body>
<div class="container">
<h1>Sleep Disorder Prediction</h1>
<form method="post">
<label>Age:<input type="number" name="Age" required></label>
<label>Sleep Duration (hours):<input type="number" step="0.1" name="Sleep Duration" required></label>
<label>Quality of Sleep (1-10):<input type="number" min="1" max="10" name="Quality of Sleep" required></label>
<label>Physical Activity Level (minutes/day):<input type="number" name="Physical Activity Level" required></label>
<label>Stress Level (1-10):<input type="number" min="1" max="10" name="Stress Level" required></label>
<label>Heart Rate (bpm):<input type="number" name="Heart Rate" required></label>
<label>Daily Steps:<input type="number" name="Daily Steps" required></label>
<label>Gender (0=Female,1=Male):<input type="number" min="0" max="1" name="Gender" required></label>
<label>Occupation:<select name="Occupation" required>
  {% for occ in occupations %}
    <option value="{{ occ }}">{{ occ }}</option>
  {% endfor %}
</select></label>
<button type="submit">Predict Sleep Disorder</button>
</form>
</div>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <style>
        body { font-family: Arial; background: #182c25; color: #f1f1f1; }
        .container { width: 600px; margin: 40px auto; background: #213d32; padding: 30px; border-radius: 10px; text-align: center;}
        .suggestion { background: #ffcd38; color: #1a362e; padding: 15px; border-radius: 8px; margin-top: 20px; font-weight: bold; }
        a { color: #ffcd38; text-decoration: none; margin-top: 30px; display: inline-block; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
<div class="container">
<h1>Prediction Result</h1>
<p><strong>Predicted Sleep Disorder Score:</strong> {{ prediction }}</p>
<div class="suggestion">{{ suggestion }}</div>
<a href="/">Try Again</a>
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    occupations = le_occ.classes_.tolist()  # occupation strings from label encoder
    if request.method == 'POST':
        occ_str = request.form['Occupation']
        occupation_enc = le_occ.transform([occ_str])[0]  # encode to int

        features = [
            int(request.form['Age']),
            float(request.form['Sleep Duration']),
            int(request.form['Quality of Sleep']),
            int(request.form['Physical Activity Level']),
            int(request.form['Stress Level']),
            int(request.form['Heart Rate']),
            int(request.form['Daily Steps']),
            int(request.form['Gender']),
            occupation_enc
        ]

        prediction = model.predict([features])[0]

        if prediction < 4:
            suggestion = "Your sleep cycle needs improvement. Consider enhancing your sleep hygiene."
        elif 4 <= prediction <= 6:
            suggestion = "Try incorporating regular exercise to improve your sleep quality."
        else:
            suggestion = "Increase your physical activity for better sleep health."

        return render_template_string(result_html, prediction=prediction, suggestion=suggestion)

    return render_template_string(home_html, occupations=occupations)

if __name__ == '__main__':
    app.run(debug=True)
