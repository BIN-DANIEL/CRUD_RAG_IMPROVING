import os
from multiprocessing import Pool
from functools import partial

import jieba
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from tqdm import tqdm
from elasticsearch.helpers import bulk

from src.retrievers.bm25 import CustomBM25Retriever


def _parse_document(document, chunk_size: int, chunk_overlap: int):
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return node_parser.get_nodes_from_documents([document])


class CustomBM25RetrieverJieba(CustomBM25Retriever):

    def __init__(
            self,
            docs_directory: str,
            embed_model=None,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs_80k",
            construct_index: bool = False,
            similarity_top_k: int = 2,
            max_workers: int = 16,
            es_host: str = 'localhost',
            es_port: int = 9221,
            es_scheme: str = 'http',
        ):
        self.max_workers = max_workers
        # 父类 __init__ 负责建立 ES 连接（construct_index=False 不触发建库）
        super().__init__(
            docs_directory,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection_name=collection_name,
            construct_index=False,
            similarity_top_k=similarity_top_k,
            es_host=es_host,
            es_port=es_port,
            es_scheme=es_scheme,
        )
        if construct_index:
            self.construct_index()

    def construct_index(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        docs_path = os.path.join(project_root, self.docs_directory)
        print("正在阅读文档...")
        documents = SimpleDirectoryReader(docs_path).load_data()
        print(f"一共加载了 {len(documents)} 个文档，开始并行切分 chunk（{self.max_workers} 进程）")

        worker = partial(
            _parse_document,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        nodes = []
        with Pool(processes=self.max_workers) as pool:
            for doc_nodes in tqdm(
                pool.imap_unordered(worker, documents, chunksize=1),
                total=len(documents),
                desc="切分文档",
            ):
                nodes.extend(doc_nodes)

        self.chunks = [node.text for node in nodes]

        # whitespace mapping：content 原文（只存不索引），content_seg 分词结果（检索用）
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text", "index": False},
                    "content_seg": {"type": "text", "analyzer": "whitespace"},
                }
            }
        }
        self.es_client.indices.create(index=self.collection_name, body=mapping, ignore=400)

        def gen_actions():
            for node in nodes:
                yield {
                    "_index": self.collection_name,
                    "_source": {
                        "content": node.text,
                        "content_seg": " ".join(jieba.cut(node.text)),
                    },
                }

        success, errors = bulk(self.es_client, gen_actions(), chunk_size=5000)
        print(f"Bulk indexed {success} chunks, {len(errors)} errors.")
        self.es_client.indices.refresh(index=self.collection_name)
        print("Indexing finished!")

    def search_docs(self, query_text: str) -> str:
        # query 也要同一套 jieba 预分词，保证和文档侧分词粒度一致
        seg_query = " ".join(jieba.cut(query_text))
        dsl = {
            "query": {"match": {"content_seg": seg_query}},
            "size": self.similarity_top_k,
        }
        search_result = self.es_client.search(index=self.collection_name, body=dsl)

        result = []
        if search_result["hits"]["hits"]:
            for record in search_result["hits"]["hits"]:
                result.append(record["_source"]["content"])  # 返回原文，而非分词结果

        return "\n".join(result)


if __name__ == "__main__":
    retriever = CustomBM25RetrieverJieba(
        docs_directory="data/80000_docs",
        chunk_size=128,
        chunk_overlap=0,
        collection_name="bm_docs_80k_chunksize_128_overlap_0_jieba",
        construct_index=True,
    )
    print(f"Chunks count: {len(retriever.chunks)}")