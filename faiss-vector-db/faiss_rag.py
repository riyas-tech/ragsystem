
import warnings
print("hello")

# Load Libraries
import os
from dotenv import load_dotenv
import numpy as np 
import warnings
warnings.filterwarnings('ignore')


#langchain core imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

#langchain specific imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

#load environment variables

load_dotenv()

#Data ingestion and processing

sample_documents = [
    Document(
        page_content="""
        Artificial Intelligence (AI) is the simulation of human intelligence in machines.
        These systems are designed to think like humans and mimic their actions.
        AI can be categorized into narrow AI and general AI.
        """,
        metadata={"source": "AI Introduction", "page": 1, "topic": "AI"}
    ),
    Document(
        page_content="""
        Machine Learning is a subset of AI that enables systems to learn from data.
        Instead of being explicitly programmed, ML algorithms find patterns in data.
        Common types include supervised, unsupervised, and reinforcement learning.
        """,
        metadata={"source": "ML Basics", "page": 1, "topic": "ML"}
    ),
    Document(
        page_content="""
        Deep Learning is a subset of machine learning based on artificial neural networks.
        It uses multiple layers to progressively extract higher-level features from raw input.
        Deep learning has revolutionized computer vision, NLP, and speech recognition.
        """,
        metadata={"source": "Deep Learning", "page": 1, "topic": "DL"}
    ),
    Document(
        page_content="""
        Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.
        It combines computational linguistics with machine learning and deep learning models.
        Applications include chatbots, translation, sentiment analysis, and text summarization.
        """,
        metadata={"source": "NLP Overview", "page": 1, "topic": "NLP"}
    )
]

#Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=[" "]
)

chunks = text_splitter.split_documents(sample_documents)


print(f"Created {len(chunks)} chunks from {len(sample_documents)} documents")

# Load the embedding models 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


#Initialize the embedding models

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions = 1536
)

## Create embedding for a single text 
sample_text = "What is machine learning  ?"
sample_embeddings = embeddings.embed_query(sample_text)

# Create FAISS vector store
vectorstore=FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print(f"Vector store created with {vectorstore.index.ntotal} vectors")

# Save vector for later use

vectorstore.save_local("faiss_index")
print("Vector store saved to 'faiss_index' directory")

# load vector store

loaded_vectorstore=FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Loaded vector store contains {loaded_vectorstore.index.ntotal} vectors")

# Similarity search
query = "What is deep learning "
results = vectorstore.similarity_search(query, k = 3)

print(f"Query : {query}\n")
print("Top 3 similar chunks")
for i , doc in enumerate(results):
    print(f"\n {i+1}, Source: {doc.metadata['source']}")
    print(f"     Content: {doc.page_content[:200]}")


### Similarity Search with score
results_with_scores=vectorstore.similarity_search_with_score(query,k=3)

print("\n\nSimilarity search with scores:")
for doc, score in results_with_scores:
    print(f"\nScore: {score:.3f}")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:100]}...")    

### Search with metadata filtering
filter_dict={"topic":"ML"}
filtered_results=vectorstore.similarity_search(
    query,
    k=3,
    filter=filter_dict
)
print(filtered_results)    

