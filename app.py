from flask import Flask, render_template, request
import pickle

#load the models
file = open("models/cv.pkl",'rb')
tokenizer= pickle.load(file) #this is a count vectorizer feature transformation (raw text --> X)
file.close()

file = open("models/clf.pkl",'rb')
model = pickle.load(file) #model trained to predict spam/no-spam based on count vectorized input 
file.close()

#Create a Flask instance 
app = Flask(__name__)

#route of the homepage
#GET request --> get the content of html page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


#the predict route
#POST request --> this route will be activated when the user clicks the button
@app.route('/predict', methods=['POST'])
def predict():
    mail_text = request.form.get("mail_text") #the name field of the textarea of the form in index.html
    mail_text_count_vectorized = tokenizer.transform([mail_text])
    prediction = int(model.predict(mail_text_count_vectorized))
    if prediction==1:
        prediction = 'SPAM'
    else:
        prediction = 'NO SPAM'
    return render_template('index.html', text=mail_text, spam_indicator=prediction)


#let's run the web application locally 
if __name__ == '__main__':
    #host=0.0.0.0 allows your app to be accessible on any IP your machine happens to have
    #debug=True will automatically reload for code changes, and it will provide useful debug information if an error occurs.
    app.run(host='0.0.0.0', debug=True)