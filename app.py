from flask import Flask,render_template, request
import pickle
import numpy as np

model = pickle.load(open("diabetes.pkl","rb"))

app = Flask(__name__)

@app.route("/")
def man():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])
def home():
    '''
    d1 = request.form["Pregnancies"]
    d2 = request.form["Glucose"]
    d3 = request.form["BloodPressure"]
    d4 = request.form["SkinThickness"]
    d5 = request.form["Insulin"]
    d6 = request.form["BMI"]
    d7 = request.form["DiabetesPedigreeFunction"]
    d8 = request.form["Age"]
    arr = np.array([[d1,d2,d3,d4,d5,d6,d7,d8]])
    pred = model.predict(arr)
    
    '''
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    #output = np.round(prediction[0],2)
    
    return render_template("after.html",data=prediction)
    

if __name__ == "__main__":
    app.run(debug=True)