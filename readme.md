# NLP Processing Data Featurization

![alt_text](images/headImage.png)

## Table of Contents
- [Introduction](#introduction)
- [EDA](#eda)
- [Data Vectorization](#data-vectorization)
    - [Bag-of-Words](#bag-of-words)
    - [Doc2Vec](#doc2vec)
- [Creating the Featurizations](#creating-the-featurizations)
    - [Bag-of-words](#bag-of-words)
    - [Doc2Vect](#doc2vect)
- [Performance](#performance)
    - [2000 Data points](#2000-data-points)
    - [20000 Data points](#20000-data-points)
- [Conclusion](#conclusion)
- [Future Direction](#future-direction)
- [Acknowledgments](#acknowledgments)

## Introduction
For this second capstone, I decided to investigate how well the different data featurization methods perform. In order to test this out, I decided to use the Amazon reviews dataset provided by Amazon because the data included many different written reviews alongside the star reviews given by each user. This allowed me to work with scalable data and to test how well the methods were performing.

[Back to Top](#table-of-contents)

## EDA

The Amazon Review data set is a collection of over 130 million reviews split up into multiple categories. For this project, I decided to just look at the first set of book reviews. After downloading the data I found that there were over 2 million rows and my computer was struggling to even run basic scripts with it. So I wrote a script to take in the data and do a stratified train test split to preserve the ratios of reviews and massively reduce the amount of data so I could write and develop code with it.

I then decided to look at the data and understand the distributions and formating I was working with. The main features I decided to work with was the reveiw_body which was each user-written review of the data and the star_rating which was the numerical value each user gave.

The review body was filled with markdown notation and other figures that could throw off any NLP model built on it so I started by removing all markdown notation from the text and then using spaCy I built my own function to remove punctuation, contractions, stop words, and URLs. With this done I lemmed all the words and but the resulting strings through tfidf vecorizor.

After this, it was time to explore the star ratings and I found the following distribution:

![alt_text](images/starting_class_distributions.png)

As you can see the data is very skewed towards high ratings with just about 2/3 of the data having a 5-star rating. After running some basic tests, I ended up finding that because of this skew, models loved to predict that no matter what the text was that a 5-star rating sounded really good.

To fix this probelm I decided on SMOTE the data. In order to do this I train test split the data and ran then applied SMOTE to the training data in order to add more ratings for the less used ratings. This produced the following distribution which I know will produce better models:

![alt_text](images/SMOTE_class_distributions.png)
<sub><sub><sub><sub><sub>much better</sub></sub></sub></sub></sub>

[Back to Top](#table-of-contents)

## Data Vectorization

Even though this project is focused on strings, we still need to vectorize the data so that the models we use to predict will understand the data its looking at. The most common way to do this is the Bag-of-Words model but there has been a new development called Doc2Words.

[Back to Top](#table-of-contents)

### Bag-of-Words

Bag-of-Words works by creating a matrix where the rows are the documents of the corpus and the columns are the vocabulary. After you build the Bag-of-Words model you tend to make it into a tfidf (Term Frequency-Inverse Document frequency) where the terms are the frequency a word appears in a document times how much information the word provides across the corpus.

[Back to Top](#table-of-contents)

### Doc2Vec

Word2Vec and Doc2Vec are recent developments in data science and NLP and I hope to show its power in this project. Word2Vec is decomposing any given word into a vector of a defined size where the larger the vector the more accurate it is. The advantages are that the vector retains document context and similar words will be represented similarly. For example in Bag-of-Words, Love and Like would be to separate values in the matrix with no relation to each other, but with Word2Vec, Love and like would be two similar vectors with high similarity. On top of this mathematical operations can be performed on the vectors such that <b>King</b> - <b>Man</b> + <b>Woman</b> should equal <b>Queen</b> on a well trained model.

![alt_text](images/Word2VecExample.png)

In order to produce a word Vector an Autoencoder Neural Network is used where the inputs are feature vectors of the document and the output is a prediction for a word. A window is moved over the document and selects the surrounding words and tries to predict a word based on the context of the words around it. For example, if we give the model the sentence <b>"the cat sat on"</b> the model would read it in as the vectors <b>[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]</b>. The window when then move over the inputs and choose a word to predict such as <b>sat</b>. The prediction would then be made off of the surrounding words in this case <b>cat</b> and <b>on</b>. After training, the weights that were found to work the best will be taken out of the autoencoder.

![alt_text](images/Doc2VecExample.png)

Doc2Vec is very similar but it adds in the additional input of a document vector. This then produces all the word vectors for the entire corpus as well as the document vector.

A good comparison for Bag-of-words and Doc2Vec is that Bag-of-words is the frequentist approach to NLP while Doc2Vec and Word2Vec are the Bayesian approaches.

[Back to Top](#table-of-contents)

## Creating the Featurizations

In order to study how well theses different featurization techniques perform I first had to figure out what tools and methods to use.

[Back to Top](#table-of-contents)

### Bag-of-words

For the Bag-of-words, I decided to just use sklearns prebuilt tools and use PCA to reduce the dimensionality. I first used the count vectorizer and then the Tfidf Transformer to set up the vectors. After this, I trained the model on the training data and then transformed the test data. After this, I looked at what the best choice for the number of components in PCA would be to speed up the calculation.

![alt_text](images/Screeplot.png)

From this scree plot, it looks like 1000 components will work best because it is capturing just about 90% of the variance.

[Back to Top](#table-of-contents)

### Doc2Vect

Here I decided to use two different systems, spaCy, and gensim. is a strong package for NLP and comes with its own general Doc2Vec model which I used to transform all my documents to vectors of a length of 300. Gensim is a package that includes tools to build your own Doc2Vec model. I personally decided to set the vector size to be 300, the window to look at the 15 words in front and behind the word, only consider words that appear more than 1 time and to train over 400 epochs (Thanks to [This](http://www.aclweb.org/anthology/W16-1609) paper on Doc2Vec and optimizing STS). I used the training set to build this model and then transformed the training and testing data.

Because you can directly compare vectors here are 5 documents along with the most and least similar documents according to spaCy:

|Original Document|Most Similar|Least Similar|
|-----------------|------------|-------------|
|Not only has Brooke McEldowney captured the pure essence of 'cat'|In the book Ginger Pye Jerry is just some normal boy who is trying to figure out how to buy a puppy.|Excelente libro, te capta, te entretiene, aprendes, te ries|
|Having worked with Brant for years and having had the pleasure of learning from him on a daily basis|This book delivers way more than one would expect. Let me explain that. First of all it is not that long of a book.|Excelente libro, te capta, te entretiene, aprendes, te ries|
|How wonderful to be able to write something for kids to do that is so appealing to read about as well. Makes scientific ideas easily acccessible.|This book is an eye-opener for those of us who like to think of themselves as rational people.|Ik heb nooit veel van pendelen gehouden, maar samen met dit boek is het een hele mooie|
|Prayer Rain is very inspirational.|Through this book you will experience the TRUE story of the Barrick family.  Their faith in God is truly inspirational.|WaWaWeeWa|
|I really like the way this book is set up. I feel like it is gonna be a good tool next year in nursing school|I've never been good at keeping a journal, but this has kept me excited and intrigued to see what the next day holds.|Este autor en general lo recomiendo ampliamente. Todos sus cuentos tienen moralejas|

Here is what I get from gensim on only 2000 data points:

|Original Document|Most Similar|Least Similar|
|-----------------|------------|-------------|
|You will think that Patti Hill has tasted salty tears as a widow!  From the very first moment, she nailed it!  Mibby is a delightful character that moves forward on faith and stubborn strength. She reminds us to embrace the quiet moments, and be prepared for the Whammies! Incorporating a group of one-of-a-kind characters, the author weaves a delightful plot.  A wonderful read.|I have been using the NutriBullet religiously for about a month now. This book is geared more towards the Vita Mix or similar blenders, however, if you divide each recipe in half they work well in the NutriBullet as well.|CUTE|
|good content|My favorite book out of all of them, glad to have found it and got it in good condition and for cheap.|Yum yum BOOK|
|I love to bake cookies.  I am new to decorating them, but this book explains so well I am sure I will be a master in no time.|This simple little book describes different methods almost easy to use. However, if you will try the methods you will be pleasantly surprised. This book trains you to go into a state of pure awareness - healing with quantum energy!|CUTE|
|I had read the Spence books in the past and was happy to find them again.  This books are like the books from the golden age of mysteries.  The victim deserves to die.  The plot is lighthearted and full of twists.|Al Burt is a Florida institution.  For many years I bought the Miami Herald on Sunday just so I could read his column.  Now, many of my favorite columns are available in something more permanent than the back page of a  magazine, and honed and polished and brought up to date to boot.  When I'm  not thumbing through it, this book rests right next to my copies of  Marjorie Kinnan Rawlings and Marjory Stoneman Douglas--Al Burt's|CUTE|
|i loved this book.  it was so interesting and very informational.  nice guy and a real family man.  enjoyed it and recommend it to all his fans.|I found this book to be amazing.  It is the only time i have been given  all the facts about endo, the treatments and the symtoms of endo.  I  strongly recommend this book to all people who suffer with endo, or friends  and family who know someone that does.|CUTE|

and according to my gensim built on 20000 data points:

|Original Document|Most Similar|Least Similar|
|-----------------|------------|-------------|
|very good recipe|just perfect|I be love it mary mcdonough be an excellent writer enjoy all the story from her life|
| I read the book in sit that be not hard to do since half of the book be vegan recipe there be not much here that he have not cover in prior book I do notice that he focus more heavily on extract vegetable oil I be glad he do since I read research paper present at the annual convention of the american oil chemist society can you believe the world have become this specialized on these be the guy that work with vegetable oil and figure out how to shove this stuff in every process food in your grocery store the health hazard of vegetable oil have prove to be even more harmful than animal fat the discovery that heat vegetable oil repeatedly there be slight cooling and heating of the oil each time basket of food be drop into the oil...|throughout the book I have record and report speech in the word of the individual interview I have choose to use the individual slang graphic description and vulgar language because they accurately reflect the way in which people think about and discuss pornography the use of sexually explicit and crude language be part of the story of how pornography be change our life to avoid such language in pornified would give less than full picture indeed the language and example use in pornified be sometimes shocking even for person like myself who be not easily startled author pamela paul in what be really an extension to an article she write for time magazine seek to show in this book that pornography be transform the way we live our life|who do not love koko and yum yum|
|love gemmell nice to have in hardback|book arrive on time It be in good condition I like the information available on book good for parent that need help for time out their kid|great book bill bring me back to my young day when it be fun to be child|
|great book great customer service|this book be gift so I have not even see it My granddaughter love it|I be love it mary mcdonough be an excellent writer enjoy all the story from her life|
|this be an awesome book for an new manager to read I buy few book when I first start to manage and do use some of the technique it offer and educational value|I read the whole book in sit It have great story that make you want to know more also have life lesson very well write and perfect for the pre teen to young teen age I hope the be sequal|I be love it mary mcdonough be an excellent writer enjoy all the story from her life|

With the spaCy model, we as humans can see some relationship between the documents which does show that Doc2Vec produces similar vectors for similar documents. Gensim, on the other hand, is a bit more random which makes a lot of sense when you think that it was trained on a much smaller set of data than spaCy likely was.

[Back to Top](#table-of-contents)

## Performance

In order to test the performance of these models, I decided to use the Random Forest Regressor and find the Mean Square Error for Bag-of-Words and Doc2Vec as well as the computational time in order to find which data vectorization method works the best. By using a grid search to tune the hyperparameters for each data factorization. My results are in the table below:

|Factorization|Bootstrap|Max Depth|Max Features|Min Samples Leaf|Min Samples Split|Number of Estimators|
|-------------|---------|---------|------------|----------------|-----------------|--------------------|
|Tfidf|False|None|sqrt|1|3|80|
|spaCy|False|None|sqrt|1|2|90|
|Gensim|False|None|sqrt|1|3|80|

Most of these are the default parameters of sklearn which in some cases makes the most sense. Max Depth tells the model the maximum depth to go to and since it is None, each decision tree will expand to pure leaves. Min samples leaf if the minimum amount of samples in each leaf of the trees and this goes hand in hand with Max depth because only having one sample in each leaf should create a pure leaf.

The hyperparameters that are changed from the default do make some amount of sense. because This is a large data set outliers would be much less likely which means bootstrapping is not needed to simulate the population so the full data set can be used to make the model. Min Samples split is the minimum amount of samples needed in a leaf to split and it does make sense that it would default to a smaller number because that helps the leafs split to the minimum samples per leaf. With the Tfidf and Gensim models, I think that the min sample split became 3 because two values ended up being very close together and tended to end in the same node producing much better results than if these two samples were to be split apart. And lastly, the number of estimators ended just shy of 100 because even though the random forest model each tree purposefully overfits the data, but if you make to many trees you start capturing the noise even though you are selecting the average results of all the trees.

[Back to Top](#table-of-contents)

### 2000 Data points

After training the predicting off of these modes I found the following results with a data set of 2000 points:

![alt_test](images/Model_MSE.png)

As you can see spaCy had the lowest MSE across all Test sets while Tfidf performed better than gensim, there is a bit of overlap between their MSE's 

![alt_test](images/Model_Time_Train.png)

From this, you can see that tfidf had the hightest time to train, While gensim and spaCy took just about the same time to train. As a point of comparison remember that both of the Doc2Vec methods produce vectors of length 300 while tfidf is 1000 items long.

What in this case I think that I would pick spaCy for data featurization because it has the best MSE and is tied for build time at lower levels.

[Back to Top](#table-of-contents)

### 20000 Data points

To See what happens with more data points I increased the amount of data and re-ran the code to see how the data would perform

![alt_text](images/Model_MSE_medium.png)

Once again the spaCy Doc2Vec is performing the best while gensim is performing the worst. and Tfidf is close to having an overlap with spaCy.

![alt_text](images/Model_Time_medium_Train.png)

And when we look at the training time, Tfidf trains the slowest by a wide margin and spacy has the fastest training time.

[Back to Top](#table-of-contents)

## Conclusion

If we look at both of these metrics and both data sets, spaCy performs the best in MSE and in training time. With gensim, it does perform the worst but from the research I have looked at, it seems that for a Doc2Vec to perform well it needs to train on a large amount of data (~ half a million). Even though I am using an out of the box model with spaCy, the model that spaCy is using has been trained on a lot more data than I have trained my gensim model on.

In conclusion, it seems that in smaller datasets spaCy perform the best but according to research building a model will perform better in the long run.

[Back to Top](#table-of-contents)

## Future Direction

I plan on taking this dataset and the tools that I have learned to start applying Doc2Vec into deep learning in order to build much stronger models on larger data. Some interesting things to look at is using Doc2Vec and review scores to make a recommendation system based on the user's written review.

[Back to Top](#table-of-contents)

## Acknowledgments

- [Taite Sandefer](https://github.com/tsandefer) For helping me understand Doc2Vec and sending me papers and some code
- The Galvanize Instructor team for helping me with code and narrowing my focus
- [The Paper](http://www.aclweb.org/anthology/W16-1609) by Jey Han Lau and Timothy Baldwin for Evaluation and setting up a Doc2Vec model
- [The Video](https://www.youtube.com/watch?v=zFScws0mb7M) From Rober Meyer explaining what Doc2vec is and what tools to use
- [This Article](https://towardsdatascience.com/yet-another-twitter-sentiment-analysis-part-1-tackling-class-imbalance-4d7a7f717d44) by Ricky Kim for an explanation and the code to clean text using spaCy
- Amazon for providing the [data set](https://registry.opendata.aws/amazon-reviews/) for this capstone

[Back to Top](#table-of-contents)
