from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import json
from tqdm import tqdm



class DenseRetrieval:
    
    def __init__(self,corpus_file, embeddings=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.ids, self.documents = self._load_documents(corpus_file)

        
    def _load_documents(self, file_path):
            
        ids = []
        documents = []
            
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Processing documents"):
                
                line = json.loads(line.strip())
                docid = line['_id']
                
                title = line['title']
                content = line['text']
                document = f"{title}\n{content}"
                documents.append(document)
                ids.append(docid)

        return ids, documents


    # Function to read the text files and encode them
    def encode_documents(self):
        batch_size = 64
        embeddings = []

        for batch_id in tqdm(range(0, len(self.documents), batch_size), desc="Encoding documents"):
            
            batch = self.documents[batch_id:batch_id + batch_size]
            
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embedding)
            
        return torch.cat(embeddings, dim=0) 

    def search(self, query, embeddings, top_k=1000, device='cpu', normalize=True):
        """
        Perform dense retrieval over precomputed document embeddings.

        Args:
            query (str): The query string.
            model: Transformer model (e.g., BERT).
            tokenizer: Corresponding tokenizer.
            embeddings (np.ndarray or torch.Tensor): Precomputed document embeddings.
            ids (list): Document identifiers corresponding to embeddings.
            top_k (int): Number of top results to return.
            device (str): 'cuda' or 'cpu'.
            normalize (bool): Whether to normalize vectors before cosine sim.

        Returns:
            List of tuples: (doc_id, similarity_score)
        """

        # Ensure embeddings are torch tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float)

        embeddings = embeddings.to(device)

        # Encode query
        self.model.eval()
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1)  # mean pooling

        if normalize:
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity
        sim_scores = torch.matmul(query_embedding, embeddings.T).squeeze(0)  # [num_docs]

        # Get top-k indices
        top_k_indices = torch.topk(sim_scores, k=top_k).indices.cpu().numpy()

        results = []
        for i in top_k_indices:
            results.append({
                'doc_id': self.ids[i],
                'score': float(sim_scores[i]),
                'text': self.documents[i]
            })

        return results




