from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer, AutoModel


import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import json
from tqdm import tqdm
import subprocess



class LLMValidation:
    
    def __init__(self,corpus_file, model_name_tokenizer, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_tokenizer)
        self.model = AutoModel.from_pretrained(model_name_tokenizer)
        self.llm = Ollama(model=model_name, thinking=False, request_timeout=360.0)
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
    
    # Function to perform a search with llm_validation
    def search_val(self, query, embeddings, top_k=5, device='cpu', normalize=True):
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

        scores = []
        responses = []

        # out results
        for i in top_k_indices:
            prompt = f"""Rate this document's relevance to the query on scale 0-2:
            0 = Not relevant
            1 = Somewhat relevant
            2 = Highly relevant

            Query: "{query}"
            Document: {self.documents[i]}

            Provide only the score (0, 1, or 2)
            Score:"""

        response = self.llm.complete(prompt, max_length=10)
        responses.append(response['response'].strip())
            
                

        return responses


    def llm_validate_document(self, query_text, doc_content):
        """Use Ollama to validate relevance of a document to a query"""

        prompt = f"""Rate this document's relevance to the query on scale 0-2:
            0 = Not relevant
            1 = Somewhat relevant
            2 = Highly relevant

            Query: "{query_text}"
            Document: {doc_content}

            Provide only the score (0, 1, or 2) and brief explanation.
            Score:"""

        response = self.call_ollama(prompt, max_length=100)

        # Extract score from response
        score = 0
        try:
            # Look for first digit in response
            for char in response:
                if char in ['0', '1', '2']:
                    score = int(char)
                    break
        except:
            score = 0

        return score, response

  