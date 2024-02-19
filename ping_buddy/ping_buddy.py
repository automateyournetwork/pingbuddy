import os
import jc
import json
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def ping_to_json(ip_address, json_path):
    command = f'ping -c 4 {ip_address}'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
    parsed_output = jc.parse('ping', result.stdout)
    wrapped_ouput = {"info": parsed_output}

    with open(json_path, 'w') as json_file:
        json.dump(wrapped_ouput, json_file)

# Message classes
class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass

class ChatWithPing:
    def __init__(self, json_path):
        self.json_path = json_path
        self.conversation_history = []
        self.load_json()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_json(self):
        # Adjust jq_schema according to the structure of ping JSON
        jq_schema = ".info[]"
        self.loader = JSONLoader(
            file_path=self.json_path,
            jq_schema=jq_schema,
            text_content=False
        )
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def chat(self, question):
        response = self.qa.invoke(question)
        self.conversation_history.append(HumanMessage(content=question))
        # Adjust the response handling as per your actual logic
        self.conversation_history.append(AIMessage(content=str(response)))
        return response

def setup_ping_page():
    """Setup page for entering IP and initiating ping."""
    st.title('Ping Buddy - Chat with Ping Results')
    ip_address = st.text_input("Enter an IP address to ping:")
    if ip_address:
        ping_button = st.button("Ping and Chat")
        if ping_button:
            json_path = "ping_results.json"  # Define a path for the JSON file
            ping_to_json(ip_address, json_path)  # Save ping results to JSON
            st.session_state['ip_address'] = ip_address
            st.session_state['json_path'] = json_path
            st.session_state['page'] = 'chat'  # Transition to chat page
            st.rerun()  # Force a rerun to recognize the session state change immediately

def chat_with_ping_page():
    """Chat page to display and interact with ping results."""
    st.title(f"Chatting with Ping Results for: {st.session_state['ip_address']}")

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithPing(st.session_state['json_path'])

    user_input = st.text_input("Ask a question about the ping results:")
    if user_input:
        if st.button("Send"):
            with st.spinner('Thinking...'):
                response = st.session_state['chat_instance'].chat(user_input)
                if isinstance(response, dict) and 'answer' in response:
                    # Display the formatted answer from the response dictionary
                    st.markdown("**Answer:**")
                    st.markdown(f"> {response['answer']}")

                    st.markdown("**Chat History:**")
                    # Assuming conversation_history is a list of Message objects
                    for message in st.session_state['chat_instance'].conversation_history:
                        if isinstance(message, HumanMessage):
                            st.markdown(f"*You:* {message.content}")
                        elif isinstance(message, AIMessage):
                            # This breaks down the AI's response into more readable chunks if needed
                            st.markdown(f"*AI:* {message.content}")
                else:
                    st.error("There was an issue processing your request.")

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'setup'
    
    if st.session_state['page'] == 'setup':
        setup_ping_page()
    elif st.session_state['page'] == 'chat':
        chat_with_ping_page()

if __name__ == "__main__":
    main()