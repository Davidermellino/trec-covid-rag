from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import json


class Evaluation:
    def __init__(
        self,
        method_name,
        retrieval_function,
        query_path,
        qrels_path,
        llm_validator,
        queries=None,
        qrels=None,
    ):
        self.method_name = method_name
        self.retrieval_function = retrieval_function
        self.query_path = query_path
        self.qrels_path = qrels_path
        self.llm_validator = llm_validator

        self.results = {
            "map": [],
            "ndcg@10": [],
            "precision@5": [],
            "precision@10": [],
            "llm_validation_MAE": [],
        }
        self.queries = self.load_queries() if queries is None else queries
        self.qrels = self.load_qrels() if qrels is None else qrels

    def load_queries(self):
        print("Loading queries...")
        queries = {}
        with open(self.query_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line)
                queries[query["_id"]] = {
                    "text": query["text"],
                    "metadata": query.get("metadata", {}),
                }
                break

        return queries

    def load_qrels(self):
        print("Loading qrels...")
        qrels = defaultdict(dict)
        qrels_df = pd.read_csv(self.qrels_path, sep="\t")

        # Filter qrels to only include documents we loaded
        filtered_qrels = []

        for _, row in qrels_df.iterrows():
            doc_id = str(row["corpus-id"])
            qrels[str(row["query-id"])][doc_id] = int(row["score"])
            filtered_qrels.append(row)

        return qrels

    def calculate_map(self, results, qrels_dict):
        """Calculate Mean Average Precision"""
        if not results:
            return 0.0

        relevant_docs = 0
        precision_sum = 0.0

        for i, result in enumerate(results):
            doc_id = result["doc_id"]
            if doc_id in qrels_dict and qrels_dict[doc_id] > 0:
                relevant_docs += 1
                precision_at_i = relevant_docs / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / relevant_docs if relevant_docs > 0 else 0.0

    def calculate_ndcg(self, results, qrels_dict, k=10):
        """Calculate Normalized Discounted Cumulative Gain at k"""
        if not results:
            return 0.0

        # Get relevance scores for retrieved documents
        relevance_scores = []
        for result in results[:k]:
            doc_id = result["doc_id"]
            relevance_scores.append(qrels_dict.get(doc_id, 0))

        if sum(relevance_scores) == 0:
            return 0.0

        # Calculate DCG
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)

        # Calculate IDCG (perfect ranking)
        ideal_relevance = sorted(
            [score for score in qrels_dict.values() if score > 0], reverse=True
        )[:k]
        if not ideal_relevance:
            return 0.0

        idcg = ideal_relevance[0]
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_precision_at_k(self, results, qrels_dict, k=10):
        """Calculate Precision at k"""
        if not results:
            return 0.0

        relevant_in_top_k = 0
        for result in results[:k]:
            doc_id = result["doc_id"]
            if doc_id in qrels_dict and qrels_dict[doc_id] > 0:
                relevant_in_top_k += 1

        return relevant_in_top_k / min(k, len(results))

    def evaluate_queries(self, embeddings):
        print(f"Evaluating {self.method_name} on {len(self.queries)} queries...")
        query_ids = [
            qid for qid in self.queries.keys() if qid in self.qrels and self.qrels[qid]
        ]
        pbar = tqdm(range(len(query_ids)))
        for query_id in query_ids:
            query_text = self.queries[query_id]["metadata"]["query"]
            qrels_dict = self.qrels.get(query_id, {})

            if not qrels_dict:
                continue

            desc = f"Query ID: {query_id} - {query_text}"
            pbar.set_description(desc)

            # Get retrieval results
            try:
                retrieved_docs = self.retrieval_function(query_text, embeddings)
            except Exception as e:
                print(f"⚠️ Error with query {query_id}: {e}")
                continue

            map_score = self.calculate_map(retrieved_docs, qrels_dict)
            ndcg_score = self.calculate_ndcg(retrieved_docs, qrels_dict, k=10)
            p5_score = self.calculate_precision_at_k(retrieved_docs, qrels_dict, k=5)
            p10_score = self.calculate_precision_at_k(retrieved_docs, qrels_dict, k=10)
            llm_validation = self.llm_validator.search_val(
                query_text, embeddings, top_k=5
            )
            llm_val_MAE = self.llm_validator.llm_validation_MAE(
                llm_validation, qrels_dict
            )

            # Salvataggio dei risultati
            self.results["map"].append(map_score)
            self.results["ndcg@10"].append(ndcg_score)
            self.results["precision@5"].append(p5_score)
            self.results["precision@10"].append(p10_score)
            self.results["llm_validation_MAE"].append(llm_val_MAE)

            tqdm.write(
                f"MAP: {map_score:.4f}, NDCG@10: {ndcg_score:.4f}, P@5: {p5_score:.4f}, P@10: {p10_score:.4f}, LLM MAE: {llm_val_MAE:.4f}"
            )

        # Calcolo medie
        avg_results = {
            metric: np.mean(scores) if scores else 0.0
            for metric, scores in self.results.items()
        }

        return avg_results
