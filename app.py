import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from htmlTemplates import css, bot_template, user_template


def get_pdf_texts(pdf_docs) -> str:
    """Extract text from multiple PDFs; tolerant to empty pages."""
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text


def get_text_chunks(raw_text: str):
    """Split text into overlapping chunks for retrieval."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    # strip empties
    return [c.strip() for c in chunks if c and c.strip()]


@st.cache_resource(show_spinner=False)
def get_embedder():
    """Cache the MiniLM embedder to avoid reloading across reruns."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},  # cosine-friendly
    )


def build_vector_store(text_chunks):
    """Build a FAISS index from text chunks using MiniLM embeddings."""
    embeddings = get_embedder()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


@st.cache_resource(show_spinner=False)
def get_llm():
    
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="conversational",   
        temperature=0.1,
        max_new_tokens=512,
    )
    
    return ChatHuggingFace(llm=endpoint)



def get_conversation_chain(vector_store):
    llm = get_llm()

    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",   
        output_key="answer",    
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )

    #
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  
        output_key="answer",           
        verbose=False,
    )
    return chain


def handle_userinput(user_question: str):
    """Run the chain and render chat bubbles."""
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    # Render entire chat history 
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )

    # (Optional) Show retrieved sources beneath the latest answer
    if "source_documents" in response and response["source_documents"]:
        with st.expander("Sources"):
            for i, d in enumerate(response["source_documents"], start=1):
                st.markdown(f"**{i}.** {d.metadata.get('source', 'Untitled')} · chars [{d.metadata.get('start_index','?')}–{d.metadata.get('end_index','?')}]")
                st.code(d.page_content[:1000], language="markdown")



# Streamlit App
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.write(css, unsafe_allow_html=True)

    # Session init
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.subheader("Chat with multiple PDFs :speech_balloon:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process PDFs first.")
        else:
            handle_userinput(user_question)

    if st.session_state.conversation is None:
        st.info("Upload PDFs and click **Process**, then ask a question.")
    elif not st.session_state.chat_history:
        st.info("Ask a question about your documents to start the chat.")


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    # 1) Extract text
                    raw_text = get_pdf_texts(pdf_docs)

                    # 2) Split into chunks
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("No extractable text found in the PDFs.")
                        return

                    # 3) Create vector store
                    vector_store = build_vector_store(text_chunks)

                    # 4) Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vector_store)

                st.success("Ready! Ask a question in the input box above.")


if __name__ == "__main__":
    main()
