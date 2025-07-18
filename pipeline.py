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
from LLM_as_validator.LLM_as_validator import LLMValidation
from evaluation import Evaluation

model_name_tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
file_path = 'trec-covid/corpus.jsonl'
embeddings_file_path = 'dense_retrieval/embeddings.npy'
queries_file_path = 'trec-covid/queries.jsonl'
qrels_file_path = 'trec-covid/qrels/test.tsv'

def load_embeddings(file_path):
    if os.path.exists(file_path):
        embeddings = np.load(file_path)
    else:
        embeddings = dense_retrieval.encode_documents()
        np.save(file_path, embeddings.numpy())
    return embeddings



if __name__ == "__main__":
    
    embeddings = load_embeddings(embeddings_file_path)

    dense_retrieval = DenseRetrieval(file_path, model_name_tokenizer)
    hyde = HyDE(model="tinyllama:latest")
    llm_validation = LLMValidation(corpus_file=file_path,model_name_tokenizer=model_name_tokenizer, model_name="tinyllama:latest")


    #EVALUATION OF RETRIEVAL METHODS

    evaluator_dense_retrieval = Evaluation(
        method_name="Dense Retrieval",
        retrieval_function=dense_retrieval.search,
        query_path=queries_file_path,
        qrels_path=qrels_file_path,
        llm_validator=llm_validation
    )
    
    evaluator_hyde = Evaluation(
        method_name="HyDE Retrieval",
        retrieval_function=hyde.hyde_retrieval,
        query_path=queries_file_path,
        qrels_path=qrels_file_path,
        llm_validator=llm_validation
    )
    
    
    average_results_dense_retrieval = evaluator_dense_retrieval.evaluate_queries(embeddings)
    average_results_hyde = evaluator_hyde.evaluate_queries(embeddings)
    
    print("\nDense Retrieval Results:")
    for metric, values in average_results_dense_retrieval.items():
        print(f"{metric}: {np.mean(values):.4f}")
        
    print("\nHyDE Retrieval Results:")
    for metric, values in average_results_hyde.items():
        print(f"{metric}: {np.mean(values):.4f}")

    #TRY RAG 
    
    prompt = "Can remdesivir reduce mortality in hospitalized COVID-19 patients?"
    
    answer = hyde.answer_with_context(prompt, embeddings=embeddings)

    print(answer)