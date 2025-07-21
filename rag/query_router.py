from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from rag.auto_merge import llm

choices = [
    ToolMetadata(description="滑动窗口，通过控制滑动窗口使块之间部分重叠", name="choice_1"),
    ToolMetadata(description="自动合并，将文档解析为多叉树形式的分层结构", name="choice_2"),
]

selector = LLMSingleSelector.from_defaults(llm=llm)

if __name__ == '__main__':
    selector_result = selector.select(
        choices, query="将文档拆分为分层结构是那种技术？"
    )
    print(selector_result.selections)

    selector_result = selector.select(
        choices, query="滑动窗口干了什么？"
    )
    print(selector_result.selections)
