# NLP Processing Models

![alt_text](images/headImage.png)

For this second capstone I decided to investigate how well the different NLP models perform. In order to test this out I decided to use the Amazon reveiw data set provided by Amazon because the data included many different writen reveiws along side the star reveiws given by each users. This alowed me to work with scalable data and to test how well my models were performing.

## EDA

The Amazon Reveiw data set is a collection of over 130 million reveiws split up into multiple catigories. For this project I decided to just look at the first set of book reveiws. After downloading the data I found that there was over 2 million rows and my ccomputer was struggling to even run basic scripts with it. So I wrote a script to take in the data and do a stratified train test split to perserve the ratios of reveiws and massivly reduce the amount of data so I could write and develope code with it.

I then decided to look at the data and understand the distributions and formating I was working with. The main features I decided to work with was the reveiw_body which was each users writen review of the data and the star_rating which was the numerical value each user gave.

The review body was filled with markdown notation and other figures that could throw off any NLP model built on it so I started by removing all markdown notation from the text and then using spacy I built my own function to remove puntuation, contractions, stop words, and urls. With this done I lemmed all the words and but the resulting strings through tfidf vecorizor.

After this it was time to explore the star ratings and I found the following distribution:

![alt_text](images/starting_class_distributions.png)

As you can see the data is very skewed towards high ratings with just about 2/3 of the data having a 5 star rating. After running some basic tests, I ended up finding that because of this skew, models loved to predict that no matter what the text was that a 5 star rating sounded really good.

To fix this probelm I decided on two seperate techniques, SMOTE and PCA. By using PCA I reduced the total dimentionality which changed the models from looking at words to looking at topics of importance which I thought would make it easier to apply an acurate SMOTE to each of the topcs

I train test split the data and ran then applied SMOTE to the training data in order to add more ratings for the less used ratings. This produced the following distribution which I know will produce better models:

![alt_text](images/SMOTE_class_distributions.png)

## Models

After the data cleaning it was time to build the models and figure out which ones performed the best. (something about which metrics to use). For each of the models I used Grid searching to figure out the best set of hyperperameters and give me the resulting model to find the models accuracy.

### Random Forest

The Grid search found that using the following features produced the lowest mean squred error and an accuracy of 65% (subject to change)

### Gradient Boosting

The Grid search found that using the following features produced the lowest mean squred error and an accuracy of 68% (subject to change)

### Naive Bayes

The Grid search found that using the following features produced the lowest mean squred error and an accuracy of 65% (subject to change)

### MLP NN

Stuff

## Conclusion