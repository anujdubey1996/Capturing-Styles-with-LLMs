import torch
import torch.nn as nn
import torch.nn.functional as F  
from itertools import combinations
import random
from random import sample
from itertools import product
import pandas as pd
import numpy as np
import random as python_random
from sklearn.model_selection import train_test_split


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using CUDA
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def train_test_splitting(df):
    authors = df['Author'].unique()

    # Initialize empty DataFrames for train and test sets
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    # Split each author's articles
    for author in authors:
        author_df = df[df['Author'] == author]
        sections = author_df['Section'].unique()

        for section in sections:
            # Filter for the author and section
            section_df = author_df[author_df['Section'] == section]

            # Check if the number of articles is too small to split 95%-5%
            # Ensure at least one article goes into the test set
            if len(section_df) <= 20:  # Adjust this threshold as needed
                test_size = 1 / len(section_df)  # Ensures at least one article is in the test set
            else:
                test_size = 0.05

            # Perform the split
            if len(section_df) > 1:  # Check if there's more than one article to split
                section_train, section_test = train_test_split(section_df, test_size=test_size, random_state=42)
            else:
                # If only one article, directly assign it to the test set
                section_train, section_test = section_df,  pd.DataFrame()

            # Append to the overall train and test DataFrames
            train_df = train_df.append(section_train)
            test_df = test_df.append(section_test)

    # Reset index
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, test_df


def generate_embedding_pairs(df, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize lists to store embeddings and labels
    pos_samples = []
    neg_samples = []
    
    authors = df['Author'].unique()
    
    # Collect positive samples for each author
    for author in authors:
        author_df = df[df['Author'] == author]
        author_embeddings = author_df['Embedding'].tolist()
        
        # Generate all possible positive pairs for the current author
        author_positive_pairs = list(combinations(author_embeddings, 2))
        
        for pair in author_positive_pairs:
            pos_samples.append((torch.tensor(pair[0], dtype=torch.float32),
                                torch.tensor(pair[1], dtype=torch.float32), 1))
    
    # Calculate the number of positive samples for each author and ensure equal negative samples
    for author in authors:
        author_df = df[df['Author'] == author]
        num_pos_samples = sum(1 for _ in combinations(author_df['Embedding'], 2))
        other_authors_df = df[df['Author'] != author]
        
        # Ensure there's a cap on the number of negative samples based on available other author articles
        max_neg_samples = min(num_pos_samples, len(other_authors_df))
        
        # Select random embeddings from other authors to pair with current author's embeddings
        selected_indices = np.random.choice(other_authors_df.index, max_neg_samples, replace=False)
        selected_other_embeddings = other_authors_df.loc[selected_indices, 'Embedding'].tolist()
        
        # Repeat embeddings of the current author to match the number of selected other embeddings
        repeated_current_embeddings = np.random.choice(author_df['Embedding'], max_neg_samples, replace=True).tolist()
        
        # Generate negative pairs
        for curr_emb, other_emb in zip(repeated_current_embeddings, selected_other_embeddings):
            neg_samples.append((torch.tensor(curr_emb, dtype=torch.float32),
                                torch.tensor(other_emb, dtype=torch.float32), 0))

    # Combine and shuffle the samples
    all_samples = pos_samples + neg_samples
    random.shuffle(all_samples)

    return all_samples

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(SiameseNetwork, self).__init__()
        
        self.dense = nn.Linear(embedding_dim, output_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward_once(self, x):
        x = F.relu(self.dense(x))
        x = x.unsqueeze(2)  # Add an extra dimension for pooling
        x = self.pooling(x).squeeze(2)  # Remove the extra dimension after pooling
        x = self.layer_norm(x)
        return x
        
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        
        # Cosine similarity as a measure of distance
        cos_sim = F.cosine_similarity(output1, output2)
        return torch.sigmoid(cos_sim)  # Apply sigmoid to cosine similarity
    
    def extract_embeddings(self, x):
        return self.forward_once(x)

def train(model, dataloader, epochs, optimizer, criterion):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for emb1, emb2, labels in dataloader:
            optimizer.zero_grad()
            predictions = model(emb1, emb2).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")



