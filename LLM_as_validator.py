from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer, AutoModel


import torch
import torch.nn.functional as F
import numpy as np


class LLMValidation:
    def __init__(self, ids, documents, model_name_tokenizer, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_tokenizer)
        self.model = AutoModel.from_pretrained(model_name_tokenizer)
        self.llm = Ollama(model=model_name, thinking=False, request_timeout=360.0)
        self.ids, self.documents = ids, documents

    # Function to perform a search with llm_validation
    def search_val(self, query, embeddings, top_k=5, device="cpu", normalize=True):
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
        inputs = self.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1)  # mean pooling

        if normalize:
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity
        sim_scores = torch.matmul(query_embedding, embeddings.T).squeeze(
            0
        )  # [num_docs]

        # Get top-k indices
        top_k_indices = torch.topk(sim_scores, k=top_k).indices.cpu().numpy()

        responses = []

        # out results
        for i in top_k_indices:
            prompt = f"""Rate this document's relevance to the query using only one digit:
            0 = Not relevant
            1 = Somewhat relevant
            2 = Highly relevant

            Query: {query}
            Document: {self.documents[i]}

            Only respond with the digit (0, 1, or 2). No explanation.
            Digit:"""

            response = self.llm.complete(prompt, max_tokens=5)
            response_text = response.text.strip()

            # Estrai solo il primo numero valido tra 0-2
            matches = [c for c in response_text if c in "012"]
            score = int(matches[0]) if matches else 0  # Default a 0 se nulla trovato
            responses.append(score)

        return [(self.ids[i], score) for i, score in zip(top_k_indices, responses)]

    def llm_validation_MAE(self, llm_results, qrels_dict):
        """
            Calculate Mean Absolute Error (MAE) between LLM results and qrels.
            
        """
        if not llm_results:
            return 0.0

        errors = []

        for doc_id, llm_score in llm_results:
            if doc_id in qrels_dict:
                true_score = qrels_dict[doc_id]
                error = abs(int(llm_score) - int(true_score))
                errors.append(error)

        return np.mean(errors) if errors else 0.0
