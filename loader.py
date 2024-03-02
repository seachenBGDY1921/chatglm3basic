
# （1-2）准备本地知识库文档目前支持 txt、docx、md、pdf 格式文件，使用Unstructured Loader类加载文件，
# 获取文本信息，loader类的使用参考https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/unstructured_file.html
#
# （3-4）对文本进行分割，将大量文本信息切分为chunks
#
# （5）选择一种embedding算法，对文本向量化
#
# （6）将知识库得到的embedding结果保存到数据库，就不用每次应用都进行前面的步骤
#
# （8-9）将问题也用同样的embedding算法，对问题向量化
#
# （10）从数据库中查找和问题向量最相似的N个文本信息
#
# （11）得到和问题相关的上下文文本信息
#
# （12）获取提示模板
#
# （13）得到输入大模型的prompt比如：问题：***，通过以下信息***汇总得到答案。
#
# （14）将prompt输入到LLM得到答案
