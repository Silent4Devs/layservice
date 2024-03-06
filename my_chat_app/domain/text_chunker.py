def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in data:
        if isinstance(doc, list):
            chunks.extend(text_splitter.split_documents(doc))
        else:
            chunks.extend(text_splitter.split_documents([doc]))

    return chunks
