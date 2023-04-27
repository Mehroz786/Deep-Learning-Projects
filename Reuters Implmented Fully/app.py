print("HEllo")
import numpy as np
from flask import Flask , request , render_template
from keras import models
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)

m = models.load_model("models/Reuters.pkl")

print("hello ")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods = ["POST"])
def predict():
    text = request.form['review']
    print("The value we got is ",text)

    results = custome_data(text)
    return  render_template("index.html",prediction_text = "This article belongs to -> {} with {} %".format(results[0],results[1]))





# list of the categories
articlelist = ["Business and financial",
"aluminum market",
"barley market",
"balance of payments",
"meat markets",
'castor-oil market',
"cocoa industry",
"coconut market",
"coffee market",
"copper market",
"corn market",
"cotton market",
"consumer price index and inflation",
'crude market',
"german currency",
"US dollar foreign",
"German currency",
"earn",
'fuel market',
'gas market',
'gross prodcution',
'gold market',
'grain market',
'oil market',
'hog market',
"housing market",
"income n earning",
"interest rates",
"industrial production",
"iron-steel",
"jet market",
"jobs market",
"lead market",
"leading economy indicators",
"livestock market",
"lumber market",
"meal-feed",
"money-fx",
"nickel market",
"orange market",
"petro-chemical",
"platinum market",
"rapeseed",
"reserves bank",
"retail",
"rice market"]

# article dictionary
article_dic = {index : value for index , value in enumerate(articlelist)}


def custome_data(text):
    max= 10000
    tok = Tokenizer(num_words=max)
    tok.fit_on_texts([text])
    s = tok.texts_to_sequences([text])
    x = vectorizing(s,dimension=max)
    p = m.predict(x)
    storing_the_index_of_highest_value = np.argmax(p)
    highes_value = p[0][storing_the_index_of_highest_value]
    print(p, highes_value, storing_the_index_of_highest_value)
    return (article_dic[storing_the_index_of_highest_value],highes_value*100)


def vectorizing(sequence , dimension = 10000):
    result = np.zeros((len(sequence),dimension))
    for i , s in enumerate(sequence):
        result[i , s] = 1
    return result


if __name__ == "__main__":
    app.run()