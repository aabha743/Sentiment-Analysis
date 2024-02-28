import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Sample text for analysis
text = "They are much better when compared to seema"

# Perform sentiment analysis
sentiment_scores = sia.polarity_scores(text)

# Determine the sentiment
if sentiment_scores['compound'] >= 0.05:
    sentiment = 'Positive'
elif sentiment_scores['compound'] <= -0.05:
    sentiment = 'Negative'
else:
    sentiment = 'Neutral'

# Print the sentiment and sentiment scores
print(f"Sentiment: {sentiment}")
print(f"Sentiment Scores: {sentiment_scores}")
