import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.legacy.postprocessor.longllmlingua import DEFAULT_INSTRUCTION_STR
from llama_index.legacy.schema import NodeWithScore, TextNode
from llmlingua import PromptCompressor

from rag.llm import llm
from rag.auto_merge import node_parser

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
# splitter = SemanticSplitterNodeParser(
#     buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
# )
compression_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-m3",
    node_parser=node_parser
)


persist_dir = "./vector_store/story_index"
if not os.path.exists(persist_dir):
    documents = SimpleDirectoryReader(
        input_files=["../doc/three_kingdoms.txt"]
    ).load_data()
    story_index = VectorStoreIndex.from_documents(
        documents, service_context=compression_context
    )
    story_index.storage_context.persist(persist_dir=persist_dir)
else:
    story_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persist_dir),
        service_context=compression_context,
    )

retriever = story_index.as_retriever(similarity_top_k=10)
question = "谁施了苦肉计？"
contexts = retriever.retrieve(question)
retrieved_nodes = [n.get_content() for n in contexts]

# 使用直接召回的片段组装Prompt
prompt = "\n\n".join(retrieved_nodes + [question])
answer = llm.complete(prompt)
print(str(answer))

llm_lingua = PromptCompressor(
    # microsoft/phi-2对中文支持不是很好
    model_name="microsoft/phi-2",
    # model_name="NousResearch/Llama-2-7b-hf",
    device_map="mps",
    model_config={},
    open_api_config={},
)

compressed_prompt = llm_lingua.compress_prompt(
    retrieved_nodes,
    instruction=DEFAULT_INSTRUCTION_STR,
    question=question,
    # target_token=300,
    rank_method="longllmlingua",

    # Set the special parameter for LongLLMLingua
    condition_in_question="after_condition",
    reorder_context="sort",
    dynamic_context_compression_ratio=0.3,  # or 0.4
    condition_compare=True,
    context_budget="+100",
)

compressed_prompt_txt = compressed_prompt["compressed_prompt"]

compressed_prompt_txt_list = compressed_prompt_txt.split("\n\n")
compressed_prompt_txt_list = compressed_prompt_txt_list[1:-1]

new_retrieved_nodes = [
    NodeWithScore(node=TextNode(text=t)) for t in compressed_prompt_txt_list
]

original_contexts = "\n\n".join(retrieved_nodes)
compressed_contexts = "\n\n".join([n.get_content() for n in new_retrieved_nodes])
original_tokens = llm_lingua.get_token_length(original_contexts)
compressed_tokens = llm_lingua.get_token_length(compressed_contexts)

print(compressed_contexts)
print("Original Tokens:", original_tokens)
print("Compressed Tokens:", compressed_tokens)
print("Compressed Ratio:", f"{original_tokens / (compressed_tokens + 1e-5):.2f}x")

# 使用压缩后的召回片段组装Prompt
compressed_contexts = compressed_contexts + "\n\n" + question
answer = llm.complete(compressed_contexts)
print(str(answer))
