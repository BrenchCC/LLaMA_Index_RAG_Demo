from llama_index.core import PromptTemplate

from rag.llm import llm


query_gen_str = """\
    你是一个专业的查询重写助手，可以根据输入的单个查询生成多个相关查询。
    总共生成 {num_queries} 个相关查询，每行一个，与以下输入查询相关。
    Query: {query}
    Queries:
"""
query_gen_prompt = PromptTemplate(query_gen_str)


def generate_queries(query: str, llm, num_queries: int = 4):
    response = llm.predict(
        query_gen_prompt, num_queries=num_queries, query=query
    )
    queries = response.split("\n")
    queries_str = "\n".join(queries)
    print(f"Generated queries:\n{queries_str}")
    return queries


if __name__ == '__main__':
    queries = generate_queries('帮我规划一下五一出行计划', llm)
    print(queries)
