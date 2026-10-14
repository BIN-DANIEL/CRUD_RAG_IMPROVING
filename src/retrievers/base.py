from abc import ABC
import os

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index import download_loader

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
import faiss
from llama_index.vector_stores import FaissVectorStore
from llama_index import load_index_from_storage


class BaseRetriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            construct_index: bool = False,
            add_index: bool = False,
            similarity_top_k: int=2,
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        # 持久化目录：src/index_vector_storage/{collection_name}
        self.persist_dir = os.path.join(
            os.path.dirname(__file__), '..', 'index_vector_storage', collection_name
        )

        if construct_index:
            self.construct_index()
        else:
            self.load_index_from_faiss()
        
        if add_index:
            self.add_index()

        # self.query_engine = self.vector_index.as_query_engine()
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k,
        )

        # assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

    def construct_index(self):
        documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        faiss_index = faiss.IndexFlatL2(self.embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.vector_index = GPTVectorStoreIndex(
            nodes, service_context=service_context, 
            storage_context=storage_context, show_progress=True
        )
        # 持久化到本地目录
        self.vector_index.storage_context.persist(persist_dir=self.persist_dir)
        print("Indexing finished!")

    def add_index(self):
        if self.docs_type == 'json':
            JSONReader = download_loader("JSONReader")
            documents = JSONReader().load_data(self.docs_directory)
        else:
            documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        faiss_index = faiss.IndexFlatL2(self.embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.vector_index = GPTVectorStoreIndex(
            nodes, service_context=service_context, 
            storage_context=storage_context, show_progress=True
        )
        self.vector_index.storage_context.persist(persist_dir=self.persist_dir)

        print("Indexing finished!")
    
    def load_index_from_faiss(self):
        vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
        storage_context = StorageContext.from_defaults(
            persist_dir=self.persist_dir,
            vector_store=vector_store,
        )
        embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
        self.vector_index = load_index_from_storage(storage_context, service_context=service_context)

    def search_docs(self, query_text: str):
        response_vector = self.query_engine.query(query_text)

        response_text_list = response_vector.response.split('\n---------------------\n')
        response_text = response_text_list[1].split("\n\n")
        response_text = "\n\n".join([text for text in response_text if not text.startswith("file_path: ")])
        
        return response_text

