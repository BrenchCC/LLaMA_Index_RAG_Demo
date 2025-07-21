from llama_index.core import SimpleDirectoryReader
from rag.utils import print_prtty_json


if __name__ == '__main__':
    reader = SimpleDirectoryReader(input_files=["../doc/three_kingdoms.pdf"])
    documents = reader.load_data()
    [x.metadata.update({'author': '罗贯中'}) for x in documents]

    print_prtty_json(documents[0].metadata)