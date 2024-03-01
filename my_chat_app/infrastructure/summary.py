import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(),override=True)
#text=file

from PyPDF2 import PdfReader

use_long_text = True

if use_long_text:
    pdf_path = "/home/normam/Descargas/LaTeX.pdf"
else:
    pdf_path = "/home/normam/Descargas/notas.pdf"

with open(pdf_path, 'rb') as file:
    reader = pdf_path.PdfReader(file)       #PyPDF2.PdfReader(file)
    text = 'notas.pdf'
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()

news_article = text

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

model_name = "babbage-002"

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    model_name=model_name
)

texts = text_splitter.split_text(news_article)

docs = [Document(page_content=t) for t in texts]
print(len(docs))



#ChatOpenAI

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.7, openai_api_key='OPENAI_API_KEY', model_name=model_name)

from langchain_core.prompts import PromptTemplate

prompt_template= """Write a concise summary of the following:

"{docs}"

CONCISE SUMMARY:"""
temperatura=0.7

prompt = PromptTemplate(template=prompt_template, input_variables=["docs"])

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens = num_tokens_from_string(news_article, model_name)
print(num_tokens)

from langchain.chains.summarize import load_summarize_chain
import textwrap
from time import monotonic


model_max_tokens = 4097
verbose = True
max_summary_length = 1000

if num_tokens < model_max_tokens:
  chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose)
else:
  chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, verbose=verbose)

start_time = monotonic()
summary = chain.run(docs)
resumen_documento= textwrap.fill(summary,width=100)

summary = summary[:max_summary_length]

print(f"Chain type: {chain.__class__.__name__}")
print(f"Run time: {monotonic() - start_time}")
print(f"Summary: {textwrap.fill(summary, width=100)}")