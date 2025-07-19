from llama_index.llms.ollama import Ollama


class HyDE:
    def __init__(self, model, dense_retrieval):
        self.llm = Ollama(model=model, thinking=False, request_timeout=360.0)
        self.dense_retrieval = dense_retrieval

    def generate_hypothetical_document(self, query_text):
        """Generate a hypothetical document that would answer the query"""
        prompt = f"""Write a scientific abstract about COVID-19 research that answers: "{query_text}"

Focus on COVID-19, SARS-CoV-2, treatments, or pandemic response. Make it sound like a real research paper abstract.

Abstract:"""

        response = self.llm.complete(prompt, max_tokens=100, temperature=0.1)
        return response.text if hasattr(response, "text") else str(response)

    def hyde_retrieval(self, query_text, embeddings, top_k=1000):
        """HyDE retrieval: generate hypothetical document, then use its embedding for retrieval"""

        # Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query_text)

        if not hypothetical_doc.strip():
            print("Error: Empty hypothetical doc, falling back to original query")
            hypothetical_doc = query_text

        # Perform search using the generated document
        results = self.dense_retrieval.search(
            query=hypothetical_doc, embeddings=embeddings, top_k=top_k
        )

        return results

    def answer_with_context(self, query_text, embeddings, top_k=5):
        """
        Generate a final answer by retrieving relevant documents and combining them with the query.
        This performs full HyDE + LLM answering.
        """
        # Step 1: Retrieve documents using HyDE
        retrieved_docs = self.hyde_retrieval(query_text, embeddings, top_k=top_k)

        if not retrieved_docs:
            print("Warning: No documents retrieved, returning default message.")
            return "No relevant documents found."

        # Step 2: Combine the query with the retrieved documents
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])

        prompt = f"""You are a helpful scientific assistant answering questions about COVID-19.

        Context:
        {context}

        Question:
        {query_text}

        Answer:"""

        # Step 3: Pass the prompt to the LLM
        response = self.llm.complete(prompt, max_tokens=300, temperature=0.2)

        return response.text if hasattr(response, "text") else str(response)
