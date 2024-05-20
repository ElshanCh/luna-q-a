# -------------------------------------------------------
# Libraries
# -------------------------------------------------------
import os
import openai

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.search.documents import SearchClient

from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

import warnings
warnings.filterwarnings("ignore")

from utils.credentials import (
    AZURE_OPENAI_SERVICE,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_GPT,
    AZURE_OPENAI_GPT_EMBED,
    AZURE_OPENAI_VERSION,
    AZURE_SEARCH_SERVICE,
    AZURE_SEARCH_SERVICE_KEY,
    AZURE_SEARCH_INDEX,
)

# -------------------------------------------------------
# OpenAI credentials
# -------------------------------------------------------
openai.api_base = AZURE_OPENAI_SERVICE
openai.api_key = AZURE_OPENAI_KEY
openai.api_version = AZURE_OPENAI_VERSION
openai.api_type = "azure"
os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_KEY


### TODO:
# - Implement Semi-structured RAG approach during indexingfor text + table documents (https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev)
class VectorStore:
    """
    A class for indexing PDF documents into an Azure Search index using OpenAI embeddings.

    Attributes:
    -----------
    pdf_path : str
        The path to the PDF document to be indexed.

    Methods:
    --------
    index_document() -> None
        Indexes the PDF document into the Azure Search index.
    """

    def __init__(self, pdf_path: str) -> None:
        """
        Initializes the VectorStore class.

        Parameters:
        -----------
        pdf_path : str
            The path to the PDF document to be indexed.
        """
        self.pdf_path = pdf_path

    def index_document(self) -> None:
        """
        Indexes the PDF document into the Azure Search index.
        """
        # Create embeddings and vectore store instances
        embeddings = OpenAIEmbeddings(engine=AZURE_OPENAI_GPT_EMBED, chunk_size=1)
        # embeddings = OpenAIEmbeddings(deployment=AZURE_OPENAI_GPT_EMBED, chunk_size=1)
        # embeddings = OpenAIEmbeddings(model=AZURE_OPENAI_GPT_EMBED, chunk_size=1)

        vector_store = AzureSearch(azure_search_endpoint=AZURE_SEARCH_SERVICE,  
                                azure_search_key=AZURE_SEARCH_SERVICE_KEY,  
                                index_name=AZURE_SEARCH_INDEX,
                                embedding_function=embeddings.embed_query)
        ### Method 1: split pdfs by page
        # # Use PDF loader to split dfs
        # loader = PyPDFLoader(self.pdf_path)
        # pages = loader.load_and_split()
        # # pages[1]
        # # Insert text and embeddings into vector store
        # vector_store.add_documents(documents=pages)

        ### Method 2: split pdfs by character
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)
        vector_store.add_documents(documents=docs)