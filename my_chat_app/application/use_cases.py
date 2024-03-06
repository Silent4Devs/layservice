from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def ask_and_get_answer(vector_store, q, k=3):
    llm = ChatOpenAI(model="text-embedding-ada-002", temperature=0.7)

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    answer = chain.run(q)
    return answer
