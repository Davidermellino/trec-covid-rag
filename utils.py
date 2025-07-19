from tqdm import tqdm

import json
import numpy as np
import os


def load_documents(file_path):
    ids = []
    documents = []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing documents"):
            line = json.loads(line.strip())
            docid = line["_id"]

            title = line["title"]
            content = line["text"]
            document = f"{title}\n{content}"
            documents.append(document)
            ids.append(docid)

    return ids, documents


def load_embeddings(file_path, encode_function=None):
    if os.path.exists(file_path):
        embeddings = np.load(file_path)
    else:
        embeddings = encode_function()
        np.save(file_path, embeddings.numpy())
    return embeddings
