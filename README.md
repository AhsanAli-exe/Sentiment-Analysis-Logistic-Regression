# 🐦 Twitter Sentiment Analysis with Logistic Regression

Hey there! 👋 Welcome to my sentiment analysis project. This is a fun little journey into the world of Natural Language Processing where we teach a machine to understand whether tweets are happy or sad (well, positive or negative to be precise).

## 🎯 What Does This Project Do?

 I am using good old logistic regression (a classic machine learning algorithm) to classify tweets as either positive 😊 or negative 😔.

The cool part? We achieved **99.5% accuracy** on our test data! Not bad for a relatively simple approach, right?

## 📁 Project Structure

### 🔧 `utils.py` - The Helper Functions
It contains three essential functions:

- **`process_tweet()`**:
  - Stock market tickers ($AAPL, $TSLA, etc.)
  - Retweet markers (RT)
  - URLs and hyperlinks
  - Hashtags (#)
  - Stopwords (the, and, is, etc.)
  - Then it stems words (running → run, happily → happi)

- **`build_freqs()`**: This creates a frequency dictionary that counts how often each word appears in positive vs negative tweets. 

- **`extract_features()`**: This converts processed tweets into numbers that our machine learning model can understand. For each tweet, it creates a feature vector with:
  - Bias term (always 1)
  - Sum of positive word frequencies
  - Sum of negative word frequencies

### 🧹 `preprocessing.ipynb` - Learning the Basics
This notebook is where I first explored tweet preprocessing. It's like my practice ground where I learned how to:
- Load the Twitter dataset
- Clean tweets step by step
- Remove unwanted elements using regex
- Tokenize text properly
- Apply stemming

It's a bit messy (as learning notebooks tend to be), but it shows the thought process behind the cleaning pipeline.

### 📊 `word_frequencies.ipynb` - Understanding the Data
Here's where I dove deep into analyzing word patterns:
- Built frequency dictionaries from the entire dataset
- Explored which words appear more in positive vs negative tweets
- Got familiar with the `build_freqs()` function
- Learned how word frequencies can be powerful features for sentiment analysis

### 🚀 `sentiment_analysis_LR.ipynb` 
A complete, well-structured notebook that covers:



**Implementation:**
- Data loading and splitting (8000 training, 2000 testing tweets)
- Feature extraction using our utility functions
- Model training with scikit-learn's LogisticRegression
- Performance evaluation and testing
- A handy prediction function for new tweets

**Results:**
- Training accuracy: 99.42%
- Test accuracy: 99.50%
- Detailed classification reports
- Examples of predictions on custom tweets

## 🛠️ How to Run This Project

### Prerequisites
You'll need these Python libraries:
```bash
pip install nltk pandas numpy scikit-learn matplotlib
```

Don't forget to download the NLTK data:
```python
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
```

### Running the Code
1. Start with `sentiment_analysis_LR.ipynb` - it's self-contained and walks you through everything
2. The notebook automatically imports functions from `utils.py`
3. Run cells in order and watch the magic happen!

### Want to Try Your Own Tweets?
The main notebook includes a `predict_sentiment()` function. Just feed it any tweet text and it'll tell you if it's positive or negative with a confidence score!

## 🤔 How It Actually Works

Here's the simple version of what happens:

1. **Tweet comes in**: "I love this sunny day! 😊"
2. **Preprocessing**: Removes emojis, converts to lowercase, stems words → ["love", "sunni", "day"]
3. **Feature extraction**: Looks up how often these words appeared in positive vs negative training tweets
4. **Prediction**: Logistic regression weighs the evidence and says "This looks positive with 92% confidence!"



Feel free to play around with the code, try different parameters, or extend it in creative ways. That's how we all learn best! 

Happy coding! 🎉

---


