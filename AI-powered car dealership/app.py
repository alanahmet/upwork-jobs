import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

def main():
    # https://streamlit.io/generative-ai

    if "history" not in st.session_state:
        st.session_state["history"] = []

    def conversational_chat(query, chain):
        """Carries out conversational chat with the AI model.

        Args:
            query (str): The user's input/query for the AI.
            chain (ConversationalRetrievalChain): The language model chain for conversation.

        Returns:
            str: The AI's response to the user's input.
        """
        
        query = f"""/
            Instruction: 
            You are a car dealership your goal sell the car as much closer as possile to the price.
            Give proper, short and briefe responce to customers if customer want to buy the car and start price negotiation with highest price.
            Do not give the price information in catalog and do not forget that your mission is to sell the car as high price as possible
            Prompt:
            {query}
        """
        result = chain({"question": query,
                        "chat_history": st.session_state["history"]})
        st.session_state["history"].append((query, result["answer"]))

        return result["answer"]

    st.set_page_config(page_title="DealerGPT") 

    # Sidebar contents
    with st.sidebar:
        st.title('🤗💬 HugChat App')
        openai_api_key = st.text_input(
            "OpenAI API Key", key="file_qa_api_key", type="password")
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
        "[View the dataset](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"

        if openai_api_key and "is_api_added" not in st.session_state:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            loader = CSVLoader(
                file_path="./data/Car information and price.csv")
            data = loader.load()
            print("data loaded")
            st.session_state["is_api_added"] = True

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(data, embeddings)

            st.session_state["chain"] = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
                retriever=vectorstore.as_retriever())

    # Generate empty lists for generated and past.
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi How may I help you?"]
    # ## past stores User's questions
    if 'past' not in st.session_state:
        st.session_state['past'] = [""]

    # Layout of input/response containers
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

    def get_text():
        """Gets the user's input text from the text_input widget.

        Returns:
            str: The user's input text.
        """
        
        input_text = st.text_input(
            "You: ", "", key="input", disabled=not openai_api_key)
        return input_text
    # Applying the user input box
    with input_container:
        user_input = get_text()

    def generate_response(prompt):
        """Generates AI response for the given prompt.

        Args:
            prompt (str): The user's input prompt for the AI.

        Returns:
            str: The AI's response to the given prompt.
        """
        #print(st.session_state["history"], prompt)
        answer = conversational_chat(prompt, st.session_state["chain"])
        response = answer
        return response

    # Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            response = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in reversed(range(len(st.session_state['generated']))):
                message(st.session_state["generated"][i], key=str(i), avatar_style="shapes")

                message(st.session_state['past'][i],
                        is_user=True, key=str(i) + '_user', avatar_style="lorelei")


if __name__ == '__main__':
    main()
