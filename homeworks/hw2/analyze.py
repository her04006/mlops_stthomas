from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import logging

logging.getLogger(__name__) # get the logger from app.py

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def load_classes():
    """Load email classes from the JSON file"""
    try:
        with open('classes.json', 'r') as file:
            data = json.load(file)
            return data.get('classes', [])
    except FileNotFoundError:
        # If file doesn't exist, create it with default classes
        default_classes = ["Work", "Sports", "Food"]
        with open('classes.json', 'w') as file:
            json.dump({"classes": default_classes}, file)
        return default_classes
        

def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response

def compute_embeddings(custom_classes=None):
    """Compute embeddings for the classes"""
    if custom_classes is None:
        custom_classes = load_classes()
    logging.info("Computing embeddings for classes: %s", custom_classes)
    embeddings = model.encode(custom_classes)
    return zip(custom_classes, embeddings)
    
def classify_email(text):
    # Encode the input text
    text_embedding = model.encode([text])[0]
    
    # Get embeddings for all classes
    classes = load_classes()
    class_embeddings = compute_embeddings(classes)
    
    # Calculate distances and return results
    results = []
    for class_name, class_embedding in class_embeddings:
        # Compute cosine similarity between text and class embedding
        similarity = np.dot(text_embedding, class_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding))
        results.append({
            "class": class_name,
            "similarity": float(similarity)  # Convert tensor to float for JSON serialization
        })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    logging.info(f"Results: \n {results}")
    
    return results