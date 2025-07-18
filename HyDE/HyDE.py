from llama_index.llms.ollama import Ollama
import os
import torch
import numpy as np
from dense_retrieval.dense_retrieval import DenseRetrieval

class HyDE:
    
    def __init__(self, model):
        self.llm = Ollama(model, thinking=False, request_timeout=360.0)
        self.dense_retrieval = DenseRetrieval(
            corpus_file='trec-covid/corpus.jsonl', 
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    
    def generate_hypothetical_document(self, query_text):
        """Generate a hypothetical document that would answer the query"""
        prompt = f"""Write a scientific abstract about COVID-19 research that answers: "{query_text}"

Focus on COVID-19, SARS-CoV-2, treatments, or pandemic response. Make it sound like a real research paper abstract.

Abstract:"""

        response = self.llm.complete(prompt, max_tokens=10)
        return response.text if hasattr(response, 'text') else str(response)
    
    def hyde_retrieval(self, query_text, embeddings, top_k=1000):
        """HyDE retrieval: generate hypothetical document, then use its embedding for retrieval"""
        print(f"Generating hypothetical document for: {query_text[:40]}...")



        # Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query_text)

        if not hypothetical_doc.strip():
            print("Error: Empty hypothetical doc, falling back to original query")
            hypothetical_doc = query_text

        print(f"Generated hypothetical document ({len(hypothetical_doc)} chars)")
        
        # Perform search using the generated document
        results = self.dense_retrieval.search(
            query=hypothetical_doc,
            embeddings=embeddings,
            top_k=top_k
        )

        return results