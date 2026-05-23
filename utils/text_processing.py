import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def preprocess_text(text, dataset_name=None):
    """
    Preprocesses text by tokenizing, removing stopwords and applying stemming.
    
    Args:
        text (str): Text to preprocess
        dataset_name (str): Name of dataset to determine language
        
    Returns:
        list: List of preprocessed tokens
    """
    # Determine language based on dataset name
    if dataset_name and ('iquique' in dataset_name.lower() or 'spanish' in dataset_name.lower()):
        language = 'spanish'
    else:
        language = 'english'  # default
    
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Get stopwords for the determined language
    stop_words = set(stopwords.words(language))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    stemmer = SnowballStemmer(language)
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens
