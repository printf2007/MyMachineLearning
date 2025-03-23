import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

os.environ['http_proxy']='127.0.0.1:7890'
os.environ['https_proxy']='127.0.0.1:7890'

os.environ["LANGCHAIN_TRACING_V3"]="true"
os.environ["LANGCHAIN_API_KEY"]="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"

#调用大语言模型
model=ChatOpenAI(model='gpt-4-turbo')
msg=[
    SystemMessage(content="请将以下的内容翻译为意大利语"),
    HumanMessage(content="你好，请问你要去哪里？")
]

result=model.invoke(msg)
print(result)