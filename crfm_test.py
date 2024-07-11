from crfm import crfmChatLLM

from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)


messages = [SystemMessage(content="Hello, how are you?"), HumanMessage(content="I'm doing well, how are you?")]
llm = crfmChatLLM(model_name=f"openai/gpt-4-0613")
response = llm.generate([messages], stop=["Q:"]).generations[0][0].text
print(response)