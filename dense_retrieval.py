from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import load_documents


class DenseRetrieval:
    def __init__(
        self, corpus_file, model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.ids, self.documents = load_documents(corpus_file)

    # Function to read the text files and encode them
    def encode_documents(self):
        
        """
        Encode documents using the pre-trained model.
        This function reads the documents, tokenizes them, and computes their embeddings.
        It returns a tensor of embeddings.
        """
        batch_size = 64
        embeddings = []

        for batch_id in tqdm(
            range(0, len(self.documents), batch_size), desc="Encoding documents"
        ):
            batch = self.documents[batch_id : batch_id + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embedding)

        return torch.cat(embeddings, dim=0)

    def search(self, query, embeddings, top_k=1000, device="cpu", normalize=True):
        """
        Perform dense retrieval over precomputed document embeddings.

        Args:
            query (str): The query string.
            embeddings (np.ndarray or torch.Tensor): Precomputed document embeddings.
            ids (list): Document identifiers corresponding to embeddings.
            top_k (int): Number of top results to return.
            device (str): 'cuda' or 'cpu'.
            normalize (bool): Whether to normalize vectors before cosine sim.

        Returns:
            List of dict: (doc_id, score, text)
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

        results = []
        for i in top_k_indices:
            results.append(
                {
                    "doc_id": self.ids[i],
                    "score": float(sim_scores[i]),
                    "text": self.documents[i],
                }
            )

        return results
