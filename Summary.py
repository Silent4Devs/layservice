import os
#import openai
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI  #
from langchain.schema import AIMessage, HumanMessage, SystemMessage    #
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate        #
from langchain.chains.summarize import load_summarize_chain     #
from langchain.docstore.document import Document    #
from langchain.text_splitter import RecursiveCharacterTextSplitter      #
from langchain.chains import LLMChain, SimpleSequentialChain
from decouple import config  
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
import chardet

# load_dotenv(find_dotenv(),override=True)
# def OPENAI_API_KEY(self):
#     return os.environ.get("OPENAI_API_KEY", "")


llm= OpenAI(openai_api_key="OPENAI_API_KEY",temperature=0.7, model_name="text-embedding-ada-002")

pdfreader=PdfReader('notas.pdf')

text='pdfreader'

for i, page in enumerate(pdfreader.pages):
    content=page.extract_text()
    if content:
        text +=content
        




chat_messages=[
    SystemMessage(content='Eres una asistente virtual llamada Lay, eres cómica, carismatica, interesante y amable, no menciones tus cualidades, que apoya encontrando información, además de dar sugerencias en base a la información que se te proporcione'),
    HumanMessage(content=f'Haz un resumen breve y conciso del texto :\n Text:{text}')
]


with open('notas.pdf','r', encoding='utf-8', errors='ignore') as f:
    text=f.read()
    
  
llm.get_num_tokens(text)  

llm(chat_messages).content 

print(llm(chat_messages).content)
  
  

generic_template='''
Haz un resumen breve y conciso del texto :{text}, 
Texto: {text}
Idioma: {language}
'''
prompts=PromptTemplate(
    input_variables=['text','language'],
    template=generic_template
    )
prompts.format(text=text,language='Español')

chain=load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    prompt=prompts,
    verbose=False
)
output_summary=chain.run(docs)


complete_prompt=prompts.format(text=text, language='Español')

llm.get_num_tokens(complete_prompt) 

llm_chain=LLMChain(llm=llm, prompt=prompts)
summary= llm_chain.run({'text':text, 'language':'Español'})

summary

# prompt_template:str='''
# Escribe un resumen breve y conciso del texto {text}, que abarque los puntos clave 
# Agrega un título al resumen , que el resumen comience con una introducción y marque 
# '''
# prompt_template_combine=PromptTemplate.from_template(template=prompt_template, input_var=['text'])

    


# print(prompt_template_combine.template)





# load_dotenv(find_dotenv(),override=True)

#  messages= [
#      SystemMessage(content='Eres Lay, una agradable e inteligente asistente virtual'),
#      HumanMessage(content=f'haz un resumen breve y conciso del texto :{text}')
#  ]
 
#  llm=ChatOpenAI(temperature=0.7, model_name="text-embedding-ada-002")
 
#  llm.(text)
 