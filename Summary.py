import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI  #
from langchain.schema import AIMessage, HumanMessage, SystemMessage    #
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate        #
from langchain.chains.summarize import load_summarize_chain     #
from langchain.docstore.document import Document    #
from langchain.text_splitter import RecursiveCharacterTextSplitter      #
from langchain.chains import LLMChain, SimpleSequentialChain
import chardet

with open('notas.pdf','r', encoding='utf-8', errors='ignore') as f:
    text=f.read()
    
llm=ChatOpenAI(temperature=0.7, model_name="text-embedding-ada-002")
   

map_prompt='''
Hazme un resumen breve y conciso del texto :{text}, 
El resumen que esté en español.
'''
map_prompt_template=PromptTemplate(
    input_variables=['text'],
    template=map_prompt
    )

prompt_complete='''
Escribe un resumen breve y conciso del texto :{text}, que abarque los puntos clave 
Agrega un título al resumen , que el resumen comience con una introducción y marque 
los puntos importantes tipo "BULLET Point" , y termina el resumen con una conclusión
Text:{text}
'''
prompt_template_combine=PromptTemplate(template=prompt_complete, input_variables=['text'])

    
chain=load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    map_prompt= map_prompt_template,
    combine_prompt=prompt_template_combine,
    verbose=False
)

summary=prompt_template_combine.run(chunks)

print(summary)





# load_dotenv(find_dotenv(),override=True)

#  messages= [
#      SystemMessage(content='Eres Lay, una agradable e inteligente asistente virtual'),
#      HumanMessage(content=f'haz un resumen breve y conciso del texto :{text}')
#  ]
 
#  llm=ChatOpenAI(temperature=0.7, model_name="text-embedding-ada-002")
 
#  llm.(text)
 