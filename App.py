from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('Pickle Module.pkl','rb'))

@app.route('/', methods = ["POST", "GET"])
def home():
    if request.method == "POST":
        inp = request.form['NOC']
        inp = int(inp)
        answer = model.predict([[inp]])
        answer = int(answer[0][0])
        return render_template('index.html', prediction_text='Predicted number of new deaths: {}'.format(answer))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()