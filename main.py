from dense_retrieval import DenseRetrieval
from LLM_as_validator import LLMValidation
from HyDE import HyDE
from evaluation import Evaluation
from utils import load_embeddings

import numpy as np


model_name_tokenizer = "sentence-transformers/all-MiniLM-L6-v2"
file_path = "trec-covid/corpus.jsonl"
embeddings_file_path = "dense_retrieval/embeddings.npy"
queries_file_path = "trec-covid/queries.jsonl"
qrels_file_path = "trec-covid/qrels/test.tsv"


def main():
    dense_retrieval = DenseRetrieval(file_path, model_name_tokenizer)
    hyde = HyDE(model="tinyllama:latest", dense_retrieval=dense_retrieval)
    llm_validation = LLMValidation(
        model_name_tokenizer=model_name_tokenizer,
        model_name="tinyllama:latest",
        documents=dense_retrieval.documents,
        ids=dense_retrieval.ids,
    )

    embeddings = load_embeddings(embeddings_file_path, dense_retrieval.encode_documents)

    # EVALUATION OF RETRIEVAL METHODS
    print("\n\n", "=" * 25, "EVALUATION", "=" * 25, "\n\n")

    evaluator_dense_retrieval = Evaluation(
        method_name="Dense Retrieval",
        retrieval_function=dense_retrieval.search,
        query_path=queries_file_path,
        qrels_path=qrels_file_path,
        llm_validator=llm_validation,
    )

    evaluator_hyde = Evaluation(
        method_name="HyDE Retrieval",
        retrieval_function=hyde.hyde_retrieval,
        query_path=queries_file_path,
        qrels_path=qrels_file_path,
        llm_validator=llm_validation,
        queries=evaluator_dense_retrieval.queries,
        qrels=evaluator_dense_retrieval.qrels,
    )

    average_results_dense_retrieval = evaluator_dense_retrieval.evaluate_queries(
        embeddings
    )
    average_results_hyde = evaluator_hyde.evaluate_queries(embeddings)

    print("\n\n", "=" * 25, "RESULTS", "=" * 25, "\n\n")

    print("\nDense Retrieval Results:")
    for metric, values in average_results_dense_retrieval.items():
        print(f"{metric}: {np.mean(values):.4f}")

    print("\nHyDE Retrieval Results:")
    for metric, values in average_results_hyde.items():
        print(f"{metric}: {np.mean(values):.4f}")

    print("\n\n", "=" * 25, "TEST", "=" * 25, "\n\n")

    # TRY RAG
    print("\nTesting RAG with HyDE and LLM Validation")
    prompt = "Can remdesivir reduce mortality in hospitalized COVID-19 patients?"

    answer = hyde.answer_with_context(prompt, embeddings=embeddings)

    print(f"Query: {prompt}")
    print(answer)


if __name__ == "__main__":
    main()
