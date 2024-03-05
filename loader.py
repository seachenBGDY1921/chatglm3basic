
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredImageLoader
from rapidocr_onnxruntime import RapidOCR


import os

work_dir = '/kaggle/ChatGLM3'
docs_path = os.path.join(work_dir, 'docs')


import numpy as np

from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter,MarkdownTextSplitter
from langchain.document_loaders import UnstructuredFileLoader,UnstructuredMarkdownLoader


embedding_model = 'text2vec-large-chinese'

#基于余弦相似性公式计算两个向量之间的相似度
def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

#加载text2vec-large-chinese模型
def load_embeddings():
    embedding_model_path = os.path.join(work_dir, embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    return embeddings

#计算两段文字的相似度
def get_embedding_sim(s1, s2, embeddings):
    embedding1 = embeddings.embed_query(s1)  #将文本转为向量
    print('embedding1: ', len(embedding1))
    embedding2 = embeddings.embed_query(s2)
    sim = get_cos_similar(embedding1, embedding2)
    print('sim of \'{0}\' and \'{1}\' is : {2}'.format(s1, s2, sim))
    return sim



#加载txt文件
def load_txt_file(txt_file):
    loader = UnstructuredFileLoader(os.path.join(work_dir, txt_file))
    docs = loader.load()
    print('txt:\n',docs[0].page_content[:100])
    return docs

#加载md文件
def load_md_file(md_file):
    loader = UnstructuredMarkdownLoader(os.path.join(work_dir, md_file))
    docs = loader.load()
    print('md:\n',docs[0].page_content[:100])
    return docs


#加载pdf文件
def load_pdf_file(pdf_file):
    loader = UnstructuredPDFLoader(os.path.join(work_dir, pdf_file))
    docs = loader.load()
    print('pdf:\n', docs[0].page_content[:100])
    return docs

# #加载jpg文件
# def load_jpg_file(jpg_file):
#     loader = UnstructuredImageLoader(os.path.join(work_dir, jpg_file))
#     docs = loader.load()
#     print('jpg:\n', docs[0].page_content[:100])
#     return docs

def load_jpg_file(jpg_file):
    ocr = RapidOCR()
    result, _ = ocr(os.path.join(work_dir,jpg_file))
    docs = ""
    if result:
        ocr_result = [line[1] for line in result]
        docs += "\n".join(ocr_result)
        print('jpg:\n', docs[:100])
    return docs

#从docs_path路径加载文件
for doc in os.listdir(docs_path):
    doc_path = f'{docs_path}/{doc}'
    if doc_path.endswith('.txt'):
        load_txt_file(doc_path)
    elif doc_path.endswith('.md'):
        load_md_file(doc_path)
    elif doc_path.endswith('.pdf'):
        load_pdf_file(doc_path)
    elif doc_path.endswith('.jpg'):
        load_jpg_file(doc_path)


#分割txt文件
def load_txt_splitter(txt_file, chunk_size=200, chunk_overlap=20):
    docs = load_txt_file(txt_file)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs

#分割md文件
def load_md_splitter(md_file, chunk_size=200, chunk_overlap=20):
    docs = load_md_file(md_file)
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs


#分割pdf文件
def load_pdf_splitter(pdf_file, chunk_size=200, chunk_overlap=20):
    docs = load_pdf_file(pdf_file)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs


#分割jpg文件
def load_jpg_splitter(jpg_file, chunk_size=200, chunk_overlap=20):
    docs = load_jpg_file(jpg_file)
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.create_documents([docs])
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs




#分割docs_path目录下的文件，并将其转为向量，放到FAISS向量数据库中
def load_vector_store(docs_path):
    split_docs = []
    for doc in os.listdir(docs_path):
        doc_path = f'{docs_path}/{doc}'
        if doc_path.endswith('.txt'):
            docs = load_txt_splitter(doc_path)
            split_docs.extend(docs)
        elif doc_path.endswith('.md'):
            docs = load_md_splitter(doc_path)
            split_docs.extend(docs)
        elif doc_path.endswith('.pdf'):
            load_pdf_splitter(doc_path)
        elif doc_path.endswith('.jpg'):
            load_jpg_splitter(doc_path)
        else:
            print('不支持的文件类型:', doc_path)
            continue
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

#从向量数据集进行内容查询
def sim_search(query, vector_store):
    #similarity_search_with_score返回相似的文档内容和查询与文档的距离分数
    #返回的距离分数是L2距离。因此，得分越低越好。
    re = vector_store.similarity_search_with_score(query)
    print('query result: ', re)