from abc import ABC

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

from langchain.schema.embeddings import Embeddings
from llama_index import QueryBundle
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class CustomBM25Retriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs_80k",
            construct_index: bool = False,
            similarity_top_k: int=2,
            es_host: str = 'localhost',
            es_port: int = 9221,
            es_scheme: str = 'http',
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.collection_name = collection_name
        self.es_host = es_host
        self.es_port = es_port
        self.es_scheme = es_scheme

        # 先建立 ES 连接（construct_index 里 bulk 写入要用到 self.es_client）
        self.es_client = Elasticsearch([{'host': self.es_host, 'port': self.es_port, "scheme": self.es_scheme}])
        print("Elasticsearch connected!")

        if construct_index:
            self.construct_index()

    def construct_index(self):
        # 【改动说明】
        # 原版走 GPTVectorStoreIndex + ElasticsearchStore，会对每个 chunk 算 embedding 存向量，
        # 但 search_docs 里用的是 ES 的 match 查询（BM25），根本没用到存进去的向量。
        # 所以这里改成：只做分块 → 把 chunk 纯文本批量写入 ES 的 content 字段。
        # ES 会用默认的 standard analyzer（对中文逐字切分）给 content 建倒排，
        # 这正是 search_docs 里 match 查询依赖的东西，也是"论文 ES BM25"的真实检索基础。
        documents = SimpleDirectoryReader(self.docs_directory).load_data()

        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

        # 用 ES 原生 bulk 批量写入，每条记录存 chunk 文本到 content 字段
        actions = (
            {"_index": self.collection_name, "_source": {"content": node.text}}
            for node in nodes
        )
        success, errors = bulk(self.es_client, actions, chunk_size=5000)
        print(f"Bulk indexed {success} chunks, {len(errors)} errors.")

        # 手动 refresh，让刚写入的数据立即可被检索
        # （ES 默认 1 秒才自动 refresh，不手动刷会导致建库后立即 search 查不到）
        self.es_client.indices.refresh(index=self.collection_name)

        print("Indexing finished!")

    def search_docs(self, query_text: str):
        query = QueryBundle(query_text)

        result = []
        dsl = {
            'query': {
                'match': {
                    'content': query.query_str
                }
            },
            "size": self.similarity_top_k
        }
        search_result = self.es_client.search(index=self.collection_name, body=dsl)
        if search_result['hits']['hits']:
            for record in search_result['hits']['hits']:
                text = record['_source']['content']
                result.append(text)
        
        result_str = '\n'.join(result)

        return result_str

