from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine

from rag.query_router_retrieval import llama_introduce_index
from rag.llm import llm


llama_introduce_retriever = llama_introduce_index.as_retriever(similarity_top_k=5)
bge_rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")
llm_rerank = LLMRerank(top_n=2, llm=llm)
llama_introduce_engine = RetrieverQueryEngine.from_args(
    llama_introduce_retriever, node_postprocessors=[bge_rerank, llm_rerank], llm=llm
)

if __name__ == '__main__':
    r = llama_introduce_engine.query("llamaindex有哪些特性？")
    print(r)
