# Fake News Detection Project Report

**Name:** Mohammed Altaweel


---

## 1. Introduction

So basically the idea of this project is to build something that can tell if a news article is fake or real. I thought this was a cool topic because fake news is literally everywhere and I wanted to see if AI can actually catch it.

The dataset I used is from Kaggle called the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). It had two files, `Fake.csv` and `True.csv`, which I combined into one big dataset with **44,898 articles**. Each article is either labeled `FAKE` or `REAL`. There are 23,481 fake ones and 21,417 real ones - not perfectly balanced but close enough that I didn't need to do anything special.

For the text, I combined the title and the article body, then cleaned everything up: made it all lowercase, removed links, numbers, and punctuation. That clean text is what I used in all three phases.

---

## 2. ML Models (Phase 1)

In this phase I tried three different machine learning models to see which one does the best:

- **Logistic Regression**
- **Random Forest** 
- **Naive Bayes**

I used TF-IDF to turn the text into numbers (top 10,000 words), then split the data 70/15/15 for train, validation, and test.

**Random Forest won** with a val F1 of **0.9959**, which was honestly way better than I expected. After doing GridSearch to tune Logistic Regression, I got a test F1 of **0.9944**. The models are really good at this task probably because real and fake news have very different writing styles.

---

## 3. Neural Network (Phase 2)

Here I built a Dense Neural Network on top of TF-IDF features. The baseline model got really high training accuracy (like 100%) but the val loss started going up after Epoch 3, which is Overfitting.

To fix it I added:
- **Dropout** (0.4) to randomly turn off neurons
- **L2 Regularization** to penalize big weights
- **EarlyStopping** so it stops before it gets worse

I also tried different learning rates and compared a frozen GloVe embedding model vs a fine-tuned one. The fine-tuned GloVe got a val F1 of **0.9717**, which is decent but still lower than the simple TF-IDF baseline. The best model overall was the Baseline TF-IDF with a test F1 of **0.9931**.

---

## 4. NLP (Phase 3)

In this phase I did a deeper NLP analysis of the dataset:

- **Word Frequency Analysis** - looked at the top 20 most common words
- **Word Clouds** - visualized what fake vs real articles talk about
- **TF-IDF + Logistic Regression** - fast and strong baseline
- **Dense Neural Network** - similar to Phase 2
- **Sentence Embeddings** - used a pretrained Sentence Transformer model

The word clouds were really interesting - fake news articles had a lot of political words like "trump", "clinton", while real ones had more neutral journalism-style language. TF-IDF still performed great here.

---

## 5. Key Findings

1. **Simple models can be surprisingly powerful** - Logistic Regression and Random Forest with TF-IDF hit over 99% F1. You don't always need a complicated neural network.

2. **Overfitting is a real problem with neural nets** - The baseline Dense Network memorized the training data. Adding Dropout + EarlyStopping helped a lot, but TF-IDF models still came out on top.

3. **Fake and real news have very different vocabulary** - The word clouds showed clear differences in the language used. This is probably why even simple bag-of-words models work so well here.

---

## 6. What's Next

If I had more time I would:

- Try **fine-tuning BERT** on this dataset — it's a much stronger language model and might catch more subtle patterns
- Test on **newer news articles** to see if the model still works on data from after 2018
![Model Comparison](model_comparison.png)