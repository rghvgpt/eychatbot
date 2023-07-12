import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.callbacks import StreamlitCallbackHandler
from PIL import Image

from my_pdf_lib import text_to_docs, parse_pdf
import os
# from db_chat import user_message, bot_message

cohere_api_key = os.environ.get('coherekey')
image = Image.open('OIP.jpg')
col1, inter_cols_pace, col2 = st.columns((1, 2, 1))
inter_cols_pace.image(image, use_column_width="auto")

c1, inter_cols, c2 = st.columns((1, 2, 1))
inter_cols.title("🔎 EY-Pedia")

"""
Feel free to ask any questions you have, and let our AI-based chatbot be your trusted guide in discovering the world of Vecmocon.
"""
###########################################

pages = None

uploaded_file = st.file_uploader(
    "**Upload a PDF file :**",
    type=["pdf"],
)
if uploaded_file:
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)

page_holder = st.empty()

prompt_template = """Role: Technical Assistant that is an expert in reading research papers and documentation

Text: {context}

Question: {question}

Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available."""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

#######################################
if pages:
    with page_holder.expander("File Content", expanded=False):
        pages
    embeddings = CohereEmbeddings(
        model="multilingual-22-12", cohere_api_key=cohere_api_key
    )
    store = Qdrant.from_documents(
        pages,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        distance_func="Dot",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I am responsible for making you smarter. Ask me something..."}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Go on ask me something?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        qa = RetrievalQA.from_chain_type(
            llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
            chain_type="stuff",
            retriever=store.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

        answer = qa({"query": prompt})
        result = answer["result"].replace("\n", "").replace("Answer:", "")
        # llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        # search = DuckDuckGoSearchRun(name="Search")
        # search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            # response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.write(result)
