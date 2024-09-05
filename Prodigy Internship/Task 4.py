######### TASK 4 #########

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

file_path = 'G:/internship 2/Task 4/twitter_validation.csv'
df = pd.read_csv(file_path)

# Assigning column names to a variable
text_column = 'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣'

# Sentiment Analysis
# Function to get the polarity of the text
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Apply the function to the correct column
df['Sentiment'] = df[text_column].apply(get_sentiment)

# Categorize sentiment
df['Sentiment_Category'] = pd.cut(df['Sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

# Visualization
# Plot the distribution of sentiment categories
plt.figure(figsize=(8,6))
sns.countplot(x='Sentiment_Category', data=df, palette='coolwarm')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# WordCloud for Positive Tweets
positive_tweets = df[df['Sentiment'] > 0][text_column]
positive_text = ' '.join(positive_tweets)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Positive Tweets')
plt.show()

# WordCloud for Negative Tweets
negative_tweets = df[df['Sentiment'] < 0][text_column]
negative_text = ' '.join(negative_tweets)
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Negative Tweets')
plt.show()

print("Analysis complete.")