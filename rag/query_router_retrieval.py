import os

from llama_index.core import VectorStoreIndex, load_index_from_storage, ServiceContext, StorageContext, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

from rag.llm import llm
from rag.window import sentence_window_engine


sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-m3",
    node_parser=SentenceWindowNodeParser.from_defaults(
        window_size=1,
        window_metadata_key="window",
        original_text_metadata_key="original_sentence",
    ),
)
text = 'llamaindex是一个用于处理和索引大规模文本数据的框架。它主要用于文本信息检索和相关任务。该框架具有以下特点和功能：\
    文本处理：llamaindex可以处理大规模文本数据，包括对文本进行清洗、分词、词干提取等预处理操作。\
    索引构建：该框架可以构建文本数据的索引，以便快速检索。索引的构建可以基于各种算法和数据结构，以满足不同的需求。\
    搜索功能：llamaindex提供了强大的搜索功能，可以根据用户的查询快速找到相关的文本数据。搜索可以支持各种检索算法和技术，以提供高效和准确的搜索结果。\
    可扩展性：llamaindex是一个可扩展的框架，可以轻松地集成和扩展到不同的系统和应用中。它提供了灵活的接口和组件，使开发人员可以根据需要定制和扩展功能。\
    高性能：由于llamaindex是针对大规模文本数据的处理和索引而设计的，因此具有良好的性能和扩展性。它可以处理大量的文本数据，并且在搜索和检索方面表现出色。'

persist_dir = "./vector_store/llama_introduce"
if not os.path.exists(persist_dir):
    llama_introduce_index = VectorStoreIndex.from_documents(
        [Document(text=text)], service_context=sentence_context
    )
    llama_introduce_index.storage_context.persist(persist_dir=persist_dir)
else:
    llama_introduce_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persist_dir),
        service_context=sentence_context,
    )
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
llama_introduce_engine = llama_introduce_index.as_query_engine(
    similarity_top_k=1, node_postprocessors=[postproc]
)

sentence_window_tool = QueryEngineTool.from_defaults(
    query_engine=sentence_window_engine,
    description="RAG技术介绍",
)
auto_merging_tool = QueryEngineTool.from_defaults(
    query_engine=llama_introduce_engine,
    description="LlamaIndex框架介绍",
)

selector = LLMSingleSelector.from_defaults(llm=llm)
query_engine = RouterQueryEngine(
    llm=llm,
    selector=selector,
    query_engine_tools=[
        sentence_window_tool,
        auto_merging_tool,
    ],
)

if __name__ == '__main__':
    r = query_engine.query("LlamaIndex是干什么的？")
    print(r)
