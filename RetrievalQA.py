# embedding：（自定义embedding）

from langchain.schema.embeddings import Embeddings

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese')
class CustomerEmbeddings(Embeddings):
    def __init__(self, tokenizer, model):
        super(CustomerEmbeddings, self).__init__()
        self.tokenizer, self.model = tokenizer, model

    def embed_documents(self, texts):
        """Embed search docs."""
        input = self.tokenizer(texts, padding=True, return_tensors='pt')
        embeddings = self.model(**input).pooled_logits

        return embeddings.detach().cpu().numpy()

    def embed_query(self, text):
        input = self.tokenizer(text, return_tensors='pt')
        embeddings = self.model(**input).pooled_logits

        return embeddings.detach().cpu().numpy()
# 1、需要实现embed_documents和embed_query方法，即自定义模型得到embedding的过程。
# embed_documents是批量输入，embed_query是单一输入。
# retriever：

from langchain.vectorstores import FAISS

embeddings = CustomerEmbeddings()
db = FAISS.from_texts(texts, embeddings, metadatas)

retriever = db.as_retriever()
# 1、metadatas可以包括被检索数据的其他需要的信息。
# RetrievalQA：

from langchain.chains import RetrievalQA
retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
res = retrievalQA.run(query="你好")