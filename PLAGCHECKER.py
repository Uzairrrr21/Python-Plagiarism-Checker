import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

def preprocess_text(text):
    """Removes special characters, converts to lowercase, and splits into words."""
    simple_text = "".join([char for char in text if char.isalnum() or char.isspace()])
    return simple_text.lower()

def tokenize_text(text):
    """Splits text into words."""
    return word_tokenize(text)

def remove_stopwords(words):
    """Removes common words (stopwords) that don't add much meaning."""
    stopwords_list = stopwords.words('english')
    return [word for word in words if word not in stopwords_list]

def calculate_similarity(doc1, doc2):
    """Calculates the similarity score between two documents based on word overlap."""
    words1 = remove_stopwords(tokenize_text(preprocess_text(doc1)))
    words2 = remove_stopwords(tokenize_text(preprocess_text(doc2)))

    counter1 = Counter(words1)
    counter2 = Counter(words2)

    matches = sum((counter1 & counter2).values())
    total_words = sum(counter1.values()) + sum(counter2.values())

    if total_words == 0:
        return 0  # Avoid division by zero
    similarity = matches / total_words

    return similarity

def print_similarity_score(doc1, doc2, similarity):
    """Prints the similarity score in a friendly way."""
    words1 = remove_stopwords(tokenize_text(preprocess_text(doc1)))
    words2 = remove_stopwords(tokenize_text(preprocess_text(doc2)))
    total_words = len(words1) + len(words2)
    print(f"How similar are the documents? They share {sum((Counter(words1) & Counter(words2)).values())} words out of {total_words} for a similarity of {similarity:.2f}")

def find_document_similarities(doc1, doc2):
    """Compares two documents to see how similar they are based on word overlap."""
    similarity = calculate_similarity(doc1, doc2)
    print_similarity_score(doc1, doc2, similarity)

# Example usage
document1 = "This is a first example. It has some info."
document2 = "This is a second example, kind of similar to the first one, right?"

find_document_similarities(document1, document2)