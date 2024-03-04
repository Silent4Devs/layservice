from langchain_community.llms import OpenAI         # import openAI from langChain
from langchain_openai import OpenAI                 # install: pip install -U langchain-community
# decouple to read .env variables(OpenAI Key)
from decouple import config                         # install: pip install python-decouple
from langchain.prompts import PromptTemplate    # import prompt template


# create the prompt/about the bot
prompt_template: str = """/
Eres una asistente virtual llamada Lay, eres cómica, carismatica, interesante y amable, no menciones tus cualidades, que apoya encontrando información, 
además de dar sugerencias en base a la información que se te proporcione y contestando preguntas: {question}. 
"""

prompt = PromptTemplate.from_template(template=prompt_template)

# format the prompt to add variable values
prompt_formatted_str: str = prompt.format(
    question="Eres Benito Juarez?")

llm = OpenAI(openai_api_key=config("OPENAI_API_KEY"))       # instantiate the OpenAI intance

prediction = llm.invoke(prompt_formatted_str)

print(prediction)