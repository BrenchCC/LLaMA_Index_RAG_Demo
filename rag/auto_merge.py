from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import Document

from rag.llm import llm

text = 'ChatGPT、Midjourney等生成式人工智能（GenAI）在文本生成、文本到图像生成等任务中表现出令人印象深刻的性能。然而，生成模型也不能避免其固有的局限性，包括产生幻觉的倾向，在数学能力弱，而且缺乏可解释性。因此，提高他们能力的一个可行办法是让他们能够与外部世界互动，以不同的形式和方式获取知识，从而提高所生成内容的事实性和合理性。\
检索增强生成（Retrieval-Augmented Generation, RAG）技术研究旨在提供更有依据、更依赖事实的信息来帮助解决生成式AI的幻觉倾向、专业力弱等固有缺陷。\
现在已有大多数教程都挑选一种或几种RAG技术并详细解释如何实现它们，而缺少一个对于RAG技术的系统化概述。本文的目的是想系统化梳理关键的高级RAG技术，并介绍它们的实现，以便于其他开发人员深入研究该技术。\
检索增强生成（又名RAG）为大语言模型提供从某些数据源检索到的信息，作为其生成答案的依据。RAG通常包括两个阶段：检索上下文相关信息和使用检索到的知识指导生成过程。基本上，RAG是通过检索算法找到的信息作为上下文，帮助大模型回答用户问询。查询和检索到的上下文都被注入到发送给LLM的提示中。'
documents = [Document(text=text)]

node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 128, 64]
)
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)
print(leaf_nodes[0].text)

nodes_by_id = {node.node_id: node for node in nodes}
parent_node = nodes_by_id[leaf_nodes[0].parent_node.node_id]
print(parent_node.text)

parent_node = nodes_by_id[parent_node.parent_node.node_id]
print(parent_node.text)


from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext


auto_merging_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-m3",
    node_parser=node_parser,
)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context=storage_context, service_context=auto_merging_context
)

automerging_index.storage_context.persist(persist_dir="./vector_store/merging_index")

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

automerging_retriever = automerging_index.as_retriever(
    similarity_top_k=5
)

retriever = AutoMergingRetriever(
    automerging_retriever,
    automerging_index.storage_context,
    verbose=True
)

auto_merging_engine = RetrieverQueryEngine.from_args(
    automerging_retriever, node_postprocessors=[], llm=llm
)

if __name__ == '__main__':
    r = auto_merging_engine.query("生成式人工智能性能强大，但也不能避免其固有的局限性。那么解决办法是什么？")
    print(r)
