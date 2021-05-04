# we are going to use Flask, a micro web framework

from flask import Flask, jsonify, request, render_template
import os
import pickle
from myclassifiers import MyKNeighborsClassifier, MySimpleLinearRegressor, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

# new_tree = MyDecisionTreeClassifier()
# new_tree.fit()

# make a Flask app
app = Flask(__name__)

# we need to add two routes (functions that handle requests)
# one for the homepage


@app.route("/", methods=["GET"])
def index():
    # return content and a status code
    return render_template("index.html", Result="None"), 200

# one for the /predict


@app.route("/", methods=["POST"])
def predict():
    gender = request.form["gender"]
    age = request.form["age"]
    hypertension = request.form["hypertension"]
    heart_disease = request.form["heart_disease"]
    ever_married = request.form["ever_married"]
    work = request.form["work"]
    residence = request.form["residence"]
    glucose = request.form["glucose"]
    bmi = request.form["bmi"]
    smoking = request.form["smoking"]
    data_array = [gender, age, hypertension, heart_disease,
                  ever_married, work, residence, glucose, bmi, smoking]

    result = int(predict_stroke(data_array)[0])
    print(result)
    result_string = 'test'
    if result == 1:
        result_string = 'Yes'
    elif result == 0:
        result_string = 'No'
    return render_template("index.html", Result=result_string), 200


def predict_stroke(instance):
    infile = open("tree.p", "rb")
    DT_classifier = pickle.load(infile)
    infile.close()
    # 2. use the tree to make a prediction
    try:
        return DT_classifier.predict([instance])  # recursive function
    except:
        return None


if __name__ == "__main__":
    # deployment notes
    # two main categories of how to deploy
    # host your own server OR use a cloud provider
    # there are lots of options for cloud providers... AWS, Heroku, Azure, DigitalOcean, Vercel, ...
    # we are going to use Heroku (Backend as a Service BaaS)
    # there are lots of ways to deploy a flask app to Heroku
    # 1. deploy the app directly as a web app running on the ubuntu "stack"
    # (e.g. Procfile and requirements.txt)
    # 2. deploy the app as a Docker container running on the container "stack"
    # (e.g. Dockerfile (creates a build specification for an image))
    # 2.A. build the docker image locally and push it to a container registry (e.g. Heroku's)
    # ** 2.B. ** define a heroku.yml and push our source code to Heroku's git repo and Heroku will build the docker image for us
    # 2.C. define a main.yml and push our source code to Github, where a Github Action builds an image and pushes it to the Heroku registry

    port = os.environ.get("PORT", 5000)
    app.run(debug=False)  # TODO: set debug to False for production
    # by default, Flask runs on port 5000
