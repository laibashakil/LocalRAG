from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []

    def split_text(self, text, max_words=100):
        sentences = text.split(".")
        chunk = ""
        for s in sentences:
            if len(chunk.split()) + len(s.split()) < max_words:
                chunk += s + ". "
            else:
                self.texts.append(chunk.strip())
                chunk = s + ". "
        if chunk:
            self.texts.append(chunk.strip())

    def index_documents(self, texts):
        chunk_size = 100
        self.texts = []
        for text in texts:
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                self.texts.append(chunk)
        
        batch_size = 32
        embeddings = []
        for i in range(0, len(self.texts), batch_size):
            batch = self.texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

    def retrieve_context(self, query, k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        context = "\n".join([self.texts[i] for i in indices[0]])
        return context
