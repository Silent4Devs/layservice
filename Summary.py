import os
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import create_react_agent, create_json_agent, create_structured_chat_agent,tool
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(find_dotenv(), override=True)


llm=ChatOpenAI(temperature=0.7, model_name='text-embedding-ada-002')
wikipedia= WikipediaAPIWrapper()

class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description


tools=[
    Tool(
        name='Wikipedia',
        func=wikipedia.run ,
        description='Useful for when you need to get information from wikipedia about a topic'
    )
    
]

llm=llm
tools= too
prompt='zero-shot-react-description'

agent_executor=create_structured_chat_agent(llm,tools,prompt)

output= agent_executor.run('Can you please provide a short summary of George Washington?')

print(output)