# src/retriever/retriever.py
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = "data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
# print(EMBEDDINGS_PATH)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logging.info(f"Loading model {MODEL_NAME}...")

logging.info(f"Embeddings path: {EMBEDDINGS_PATH}")

print(f"Embeddings path: {EMBEDDINGS_PATH}")

# Load and initialize model
model = SentenceTransformer(MODEL_NAME)

# Global variables for data and embeddings
df = None
embeddings = None

def load_data():
    """Load the dataset and create embeddings if they don't exist."""
    global df, embeddings
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    # Read and combine all CSVs
    all_dfs = []
    for file in csv_files:
        data_path = os.path.join(DATA_DIR, file)
        df = pd.read_csv(data_path)
        if 'wikipedia_excerpt' not in df.columns:
            pass
        all_dfs.append(df)
        print(f"Loaded data from {data_path}")

    # Concatenate all DataFrames
    df = pd.concat(all_dfs, ignore_index=True)

    df = pd.read_csv(data_path)
    print(f"Loaded data from {data_path}")
    
    # Check if embeddings already exist
    if os.path.exists(EMBEDDINGS_PATH):
        print("Loading existing embeddings...")
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        # Use PyTorch-based encoding with batching to avoid memory issues
        excerpts = df['wikipedia_excerpt'].tolist()
        batch_size = 32
        with torch.no_grad():
            all_embeddings = []
            for i in range(0, len(excerpts), batch_size):
                batch = excerpts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(excerpts) - 1)//batch_size + 1}")
                batch_embeddings = model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
def get_similar_responses(question: str, top_k=5) -> list:
    """Get similar responses to the question based on embeddings."""
    global df, embeddings
    
    # Load data if not loaded yet
    if df is None or embeddings is None:
        load_data()
    
    # Create embedding for the question
    question_embedding = model.encode(question)
    
    # Calculate similarity
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    
    # Get top_k similar indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return the excerpts corresponding to the top indices
    similar_responses = df.iloc[top_indices]['wikipedia_excerpt'].tolist()
    return similar_responses