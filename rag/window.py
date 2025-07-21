from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document

from rag.llm import llm
from rag.utils import print_prtty_json

window_node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=1,
    window_metadata_key="window",
    original_text_metadata_key="original_sentence",
)

text = ("随着最近在RAG领域的进展,Advanced RAG已经演变成一个新的范式. "
        "它对传统Native RAG范式的一些局限性进行了针对性的增强. "
        "Advanced RAG流程可以分为预检索、检索和检索后处理三个部分. "
        "检索阶段的目标是识别最相关的上下文, 通常检索过程是基于向量进行搜索的.")
documents = [Document(text=text)]
nodes = window_node_parser.get_nodes_from_documents(documents)
print_prtty_json(nodes[2].metadata)

import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage

sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-m3",
    node_parser=window_node_parser,
)

persist_dir = "./vector_store/sentence_index"
if not os.path.exists(persist_dir):
    sentence_index = VectorStoreIndex.from_documents(
        documents, service_context=sentence_context
    )
    sentence_index.storage_context.persist(persist_dir=persist_dir)
else:
    sentence_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persist_dir),
        service_context=sentence_context,
    )

from llama_index.legacy.postprocessor import MetadataReplacementPostProcessor

postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

sentence_window_engine = sentence_index.as_query_engine(
    similarity_top_k=2, node_postprocessors=[postproc]
)

if __name__ == '__main__':
    resp = sentence_window_engine.query("RAG范式有哪几种？检索过程是基于什么？")
    print(resp)
