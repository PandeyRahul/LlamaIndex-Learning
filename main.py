# Essential Library Imports
from llama_index.core import Settings
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
from llama_index.llms.ollama.base import Ollama

# Using print statements as logs
print("INFO ==> Data Loading Started...")

# Initializing the Directory Reader to process all the files from a directory
documents = SimpleDirectoryReader("data")

# Loading the documents and displaying the progress
documents = documents.load_data(show_progress=True)
print("INFO ==> Data Loaded Successfully...")

# Using Settings to add embeddings and using default embedding of LlamaIndex
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Using Settings to add the LLM model
Settings.llm = Ollama(model="llama3.2:1b", base_url="http://127.0.0.1:11434", request_timeout=360)

# Index the raw data
index = VectorStoreIndex.from_documents(documents)
print("INFO ==> Indexing Completed...")

# Creating a Query Engine
query_engine = index.as_query_engine()

# Generating response from the Query Engine
print("Bot: Please ask me a question related to the data uploaded.")
query = input("You:")
response = query_engine.query(query)

# Displaying the response
print(f"Bot: {response}")