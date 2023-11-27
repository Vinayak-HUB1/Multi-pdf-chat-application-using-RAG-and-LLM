import streamlit as st
from haystack.nodes import PDFToTextConverter, PreProcessor
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Pipeline
from haystack.nodes import PromptTemplate,AnswerParser,PromptNode
from haystack.utils import print_answers


document_store = WeaviateDocumentStore(
        host='http://localhost',
        port=8080,
        embedding_dim=384
    )

converter = PDFToTextConverter(valid_languages=["en"])

api_key = api_key

prompt_template = PromptTemplate(prompt = """"Given the provided Documents, answer the Query. Make your answer detailed and long upto 100 words \n
                                                Query: {query}\n
                                                Documents: {join(documents)}
                                                Answer: 
                                            """,
                                            output_parser=AnswerParser())

prompt_node = PromptNode(model_name_or_path = "gpt-3.5-turbo",
                            api_key = api_key,
                            max_length=1000,
                            default_prompt_template = prompt_template)


preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
)

def main():
    
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        all_docs = []
        for file in pdf_docs:
            file_ = converter.convert(file_path=file, meta=None)[0]
            all_docs.append(file_)

    docs_default = preprocessor.process(all_docs)
    print(f"n_docs_input: 1\nn_docs_output: {len(docs_default)}")

    # Not respecting sentence boundary vs respecting sentence boundary

    preprocessor_nrsb = PreProcessor(split_respect_sentence_boundary=False)
    docs_nrsb = preprocessor_nrsb.process(all_docs)
    document_store.write_documents(docs_nrsb)
    retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )

    document_store.update_embeddings(retriever)

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

    #results = query_pipeline.run('what are the name of candidates?')
    #results = query_pipeline.run('what are the skills of Vinayak?')
    if st.button("Process"):
        results = query_pipeline.run(user_question)

        #print(results['answers'][0].answer)
        sentences = results['answers'][0].answer.split('\n')
        result = '\n'.join(sentences[1:])
        print(results)
        st.info(result)


main()