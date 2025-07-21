import os

from llama_index.llms.gemini import Gemini

os.environ['all_proxy'] = 'socks5://127.0.0.1:1086'  # python3 -m pip install pysocks

llm = Gemini(model_name="models/gemini-pro", api_key=os.environ['gemini_api_key'], transport="rest")

if __name__ == '__main__':
    r = llm.complete("巴黎奥运会什么时候举办？")
    print(r)
