# NLP Processing Models

![alt_text](images/headImage.png)

For this second capstone I decided to investigate how well the different data vectorization methods perform. In order to test this out I decided to use the Amazon reveiw data set provided by Amazon because the data included many different writen reveiws along side the star reveiws given by each users. This alowed me to work with scalable data and to test how well the methods were performing.

## EDA

The Amazon Reveiw data set is a collection of over 130 million reveiws split up into multiple catigories. For this project I decided to just look at the first set of book reveiws. After downloading the data I found that there was over 2 million rows and my ccomputer was struggling to even run basic scripts with it. So I wrote a script to take in the data and do a stratified train test split to perserve the ratios of reveiws and massivly reduce the amount of data so I could write and develope code with it.

I then decided to look at the data and understand the distributions and formating I was working with. The main features I decided to work with was the reveiw_body which was each users writen review of the data and the star_rating which was the numerical value each user gave.

The review body was filled with markdown notation and other figures that could throw off any NLP model built on it so I started by removing all markdown notation from the text and then using spacy I built my own function to remove puntuation, contractions, stop words, and urls. With this done I lemmed all the words and but the resulting strings through tfidf vecorizor.

After this it was time to explore the star ratings and I found the following distribution:

![alt_text](images/starting_class_distributions.png)

As you can see the data is very skewed towards high ratings with just about 2/3 of the data having a 5 star rating. After running some basic tests, I ended up finding that because of this skew, models loved to predict that no matter what the text was that a 5 star rating sounded really good.

To fix this probelm I decided on SMOTE the data. In order to do this I train test split the data and ran then applied SMOTE to the training data in order to add more ratings for the less used ratings. This produced the following distribution which I know will produce better models:

![alt_text](images/SMOTE_class_distributions.png)

## Data Vectorization

Even though this project is focused on strings, we still need to vectorize the data so that the models we use to predict will understand the data its looking at. The most common way to do this is the Bag-of-Words model but there has been a new development called Doc2Words.

### Bag-of-Words

Bag-of-Words works by creating a matrix where the rows are the documents of the corpus and the columns are the vocabulary. After you build the Bag-of-Words model you tend to make it into a tfidf (Term Frequency Inverse Document frequency) matrix.

### Doc2Vec

Word2Vec and Doc2Vec are recent developments in data science and NLP and I hope to show its power in this project. Word2Vec is decomposing any given document into a vector of a defined size where the larger the vector the more accurate it is. The advantages is that the vector retains document context and simular words will be represented simularly. For example in Bag-of-Words, Love and Like would be to seperate values in the matrix with no relation to eachother, but with Word2Vec, Love and like would be two simular vectors with a high simularity.

![alt_text](images/Word2VecExample.png)

In order to produce a word Vector an Autoencoder Neural Network is used where the inputs are feature vectors of the document and the output is a prediction for a word. Simular to Convolutional Neural Networks a window is moved over the document and selects the suronding words and trys to predict a word based on the context of the words around it. After training is done the weights become the Word Vectors.

![alt_text](images/Doc2VecExample.png)

Doc2Vec is very simular but it adds in the additional input of a document vector which as a result produces a vector for the entire Document and not just the words in the Document. (check with Taite about explanation)

## Performance

In order to test the performance of these models I decided to use the Random Forest Regressor and find the Mean Square Error for Bag-of-Words and Doc2Vec as well as the computational time in order to find which data vectorization method works the best

## Conclusion