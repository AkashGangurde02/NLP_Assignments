# Perform tokenization (Whitespace, Punctuation-based, Treebank, Tweet, MWE) using NLTK 
# library. Use porter stemmer and snowball stemmer for stemming. Use any technique for 
# lemmatization.  

#Stemming - A technique in NLP where words are reduced to their root or vase form , also called a stem by removing suffizes and prefixes
  #Purpose - To reduce the number od the unique words in a text by grouping different inflected forms of a word under a common stem
  #Ex - Walking, Walked Walks will sstem to "Walk"

#Porter Stemmer - Classifies each character in a word as a consonant or vowel 
# Compares each word to a list of rules that specify which ending characters to remove 
# Applies a series of rules to remove suffixes and transform words to their base form  

#1. WhitespaceTokenizer - Splits tokens based on spaces (whitespace characters). Does not split punctuation
# 2. WordPunctTokenizer - Splits words at whitespace and punctuation. Useful for separating words and punctuation separately.
# 3. TreebankWordTokenizer - Uses rules based on the Penn Treebank corpus. Handles contractions, quotes, and hyphenated words intelligently.
# 4 . TweetTokenizer - Designed for social media text (tweets, messages, etc.). Handles hashtags, emojis, mentions, and special symbols better.
# 5 . MWETokenizer (Multi-Word Expression Tokenizer) - Used to group specific words as a single token. Requires predefined multi-word expressions (MWEs).


import nltk
from nltk.tokenize import (WhitespaceTokenizer, WordPunctTokenizer, 
                           TreebankWordTokenizer, TweetTokenizer, MWETokenizer)
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer #WordNet Lemmatizer from NLTK is used for lemmatization. The specific technique applied is verb-based lemmatization using wordnet.VERB.
from nltk.corpus import wordnet #WordNet is a database of English words that groups words into related sets and provides semantic relations between them

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample text
# text = "The quick brown fox jumps over the lazy dog. He's running very fast! #wildlife"

#Taking Input Manually 
text = input("Enter your sentence: ")

# Tokenization
whitespace_tokenizer = WhitespaceTokenizer()
punctuation_tokenizer = WordPunctTokenizer()
treebank_tokenizer = TreebankWordTokenizer()
tweet_tokenizer = TweetTokenizer()
mwe_tokenizer = MWETokenizer([('lazy', 'dog')])  # Example MWE

whitespace_tokens = whitespace_tokenizer.tokenize(text)
punctuation_tokens = punctuation_tokenizer.tokenize(text)
treebank_tokens = treebank_tokenizer.tokenize(text)
tweet_tokens = tweet_tokenizer.tokenize(text)
mwe_tokens = mwe_tokenizer.tokenize(text.split())

# Stemming
porter = PorterStemmer()
snowball = SnowballStemmer(language='english')

porter_stems = [porter.stem(word) for word in punctuation_tokens]
snowball_stems = [snowball.stem(word) for word in punctuation_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, wordnet.VERB) for word in punctuation_tokens]

# Output results can be shown by below code same like as the text file output , uncomment below to print it in terminal
# print("Whitespace Tokenization:", whitespace_tokens)
# print("Punctuation-based Tokenization:", punctuation_tokens)
# print("Treebank Tokenization:", treebank_tokens)
# print("Tweet Tokenization:", tweet_tokens)
# print("MWE Tokenization:", mwe_tokens)
# print("Porter Stemmer:", porter_stems)
# print("Snowball Stemmer:", snowball_stems)
# print("Lemmatization:", lemmas)

with open('assign_1.txt', 'a') as assign:
    assign.write("Your input was: " + text + "\n")
    assign.write("Whitespace Tokenization: " + str(whitespace_tokens) + "\n")
    assign.write("Punctuation-based Tokenization: " + str(punctuation_tokens) + "\n")
    assign.write("Treebank Tokenization: " + str(treebank_tokens) + "\n")
    assign.write("Tweet Tokenization: " + str(tweet_tokens) + "\n")
    assign.write("MWE Tokenization: " + str(mwe_tokens) + "\n")
    assign.write("Porter Stemmer: " + str(porter_stems) + "\n")
    assign.write("Snowball Stemmer: " + str(snowball_stems) + "\n")
    assign.write("Lemmatization: " + str(lemmas) + "\n\n")



