"""
This script is a simple web demo based on Streamlit, showcasing the use of the ChatGLM3-6B model. For a more comprehensive web demo,
it is recommended to use 'composite_demo'.

Usage:
- Run the script using Streamlit: `streamlit run web_demo_streamlit.py`
- Adjust the model parameters from the sidebar.
- Enter questions in the chat input box and interact with the ChatGLM3-6B model.

Note: Ensure 'streamlit' and 'transformers' libraries are installed and the required model checkpoints are available.
"""

import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

st.set_page_config(
    page_title="ChatGLM3-6B Streamlit Simple Demo",
    page_icon=":robot:",
    layout="wide"
)


from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from service.chatglm_service import ChatGLMService
from knowledge_service import KnowledgeService


@st.cache_resource
def get_model():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
    return tokenizer, model
class LangChainApplication():

    def __init__(self):

        self.llm_service = ChatGLMService()

        self.llm_service.load_model()

        self.knowledge_service = KnowledgeService()
    # 获取大语言模型返回的答案（基于本地知识库查询）
    def get_knowledeg_based_answer(self, query,
                                   history_len=5,
                                   temperature=0.1,
                                   top_p=0.9,
                                   top_k=4,
                                   chat_history=[]):
        # 定义查询的提示模板格式：
        prompt_template = '''
基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知内容:
{context}
问题:
{question}
    '''
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        self.llm_service.history = chat_history[-history_len:] if history_len > 0 else []
        self.llm_service.temperature = temperature
        self.llm_service.top_p = top_p

        # 利用预先存在的语言模型、检索器来创建并初始化BaseRetrievalQA类的实例
        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm_service,
            # 基于本地知识库构建一个检索器，并仅返回top_k的结果
            retriever=self.knowledge_service.knowledge_base.as_retriever(
                search_kwargs={"k": top_k}),
            prompt=prompt)
        # combine_documents_chain的作用是将查询返回的文档内容（page_content）合并到一起作为prompt中context的值
        # 将combine_documents_chain的合并文档内容改为{page_content}

        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")

        # 返回结果中是否包含源文档
        knowledge_chain.return_source_documents = True

        # 传入问题内容进行查询
        result = knowledge_chain({"query": query})
        return result

    # 获取大语言模型返回的答案（未基于本地知识库查询）
    def get_llm_answer(self, query):
        result = self.llm_service._call(query)
        return result


application = LangChainApplication()
result1 = application.get_llm_answer('比赛要求是什么？')
print('\nresult of ChatGLM3:\n')
print(result1)
print('\n#############################################\n')

application.knowledge_service.init_knowledge_base()
result2 = application.get_knowledeg_based_answer('比赛要求是什么？')
print('\n#############################################\n')
print('\nresult of knowledge base:\n')
print(result2)
# ---------------------------------------
# 加载Chatglm3的model和tokenizer


if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.chat_input("请输入您的问题")
if prompt_text:
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values

    model = application.llm_service.model
    tokenizer = application.llm_service.tokenizer

    for response, history, past_key_values in model.stream_chat(
            tokenizer,
            prompt_text,
            history,
            past_key_values=past_key_values,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature,
            return_past_key_values=True,
    ):
        message_placeholder.markdown(response)
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values
