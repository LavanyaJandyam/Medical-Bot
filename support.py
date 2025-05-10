import logging
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class getdata:
    def __init__(self):
        self.documents = None
        self.split_docs = None
        self.embeddings = None
    def load_data(self):
        loader = PyPDFLoader("The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
        self.documents = loader.load()

    def split_data(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.split_docs = splitter.split_documents(self.documents)
    # ... load_data and split_data as before ...

    def vector_embedding(self):
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [doc.page_content for doc in self.split_docs]
        self.embeddings = embedding_model.encode(texts)
        return texts

    def create_chroma_db(self):
        logging.info("Starting ChromaDB creation...")
        texts = self.vector_embedding()

        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        logging.info("Connected to ChromaDB at ./chroma_db")

        collection = chroma_client.get_or_create_collection(name="medical_chatbot")
        logging.info("Collection 'medical_chatbot' ready")

        for i, text in enumerate(texts):
            logging.info(f"Storing document ID: {i}")
            collection.add(
                ids=[str(i)],
                documents=[text],
                embeddings=[self.embeddings[i].tolist()],
            )

        logging.info("✅ Documents stored in ChromaDB successfully!")

data_loader = getdata()
data_loader.load_data()
data_loader.split_data()
data_loader.create_chroma_db()