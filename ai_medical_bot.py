import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

faiss_path = "/Users/resafroshanpm/Library/CloudStorage/OneDrive-ChakrInnovations/projects/M_C/vectorstore/db_faiss"
vectorstore = FAISS.load_local(
    folder_path=faiss_path,
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True 
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

llm = Ollama(
    model="mistral",
    temperature=0.3, 
    system="You are a helpful and knowledgeable medical assistant. Always provide factual, clear, and safe medical guidance."
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

st.set_page_config(page_title="Medical Chatbot", page_icon="💬")
st.title("🩺 Medical Assistant Chatbot")
st.write("This chatbot uses **Mistral LLM + FAISS** to provide safe, factual medical guidance.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your medical question:")

if query:
    with st.spinner("Thinking..."):
        result = qa({"question": query})
        answer = result["answer"]

        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))

for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
