from transformers import AutoTokenizer, AutoModel
import torch
import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict


from dense_retrieval.dense_retrieval import DenseRetrieval
from HyDE.HyDE import HyDE
from evaluation import Evaluation

model_name = "sentence-transformers/all-MiniLM-L6-v2"
file_path = 'trec-covid/corpus.jsonl'
embeddings_file_path = 'dense_retrieval/embeddings.npy'
queries_file_path = 'trec-covid/queries.jsonl'
qrels_file_path = 'trec-covid/qrels/test.tsv'


if __name__ == "__main__":
    

    dense_retrieval = DenseRetrieval(file_path, model_name)
    hyde = HyDE(model="llama3.2:3b")

    if os.path.exists(embeddings_file_path):
        embeddings = np.load(embeddings_file_path)
    else:
        embeddings = dense_retrieval.encode_documents()
        np.save(embeddings_file_path, embeddings.numpy())

    evaluator = Evaluation(
        method_name="Dense Retrieval",
        retrieval_function=dense_retrieval.search,
        query_path=queries_file_path,
        qrels_path=qrels_file_path
    )
    
    
    average_results = evaluator.evaluate_queries( embeddings)
    
    for metric, values in average_results.items():
        print(f"{metric}: {np.mean(values):.4f}")
    #results = evaluate_queries('trec-covid/queries.jsonl', embeddings, dense_retrieval_model=dense_retrieval)

    print("Testing HyDE retrieval...")
    test_hyde_results = hyde.hyde_retrieval("what is the origin of COVID-19",embeddings, top_k=3)

    print(f"Hypothetical document:")
    print(f"{test_hyde_results[0]['hypothetical_doc'][:200]}...")
    print(f"\nTop 3 HyDE results:")
    for i, result in enumerate(test_hyde_results):
        print(f"{i+1}. Score: {result['score']:.3f}")
        print(f"   Title: {result['title'][:80]}...")
        print()