import streamlit as st
import requests
import re

API_URL = "https://api-inference.huggingface.co/models/"
API_TOKEN = "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"  # Replace this with your actual API token
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def split_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)


def is_similar_to_input(sentence, user_input):
    return user_input.lower() in sentence.lower()


def get_response(user_input, model_name, temperature=0.9, top_p=0.9):
    payload = {
        "inputs": user_input,
        "options": {
            "max_length": 50,
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 2,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequence": "\\n",
        },
    }
    with st.spinner("Connecting... Please wait."):
        response = requests.post(API_URL + model_name, headers=headers, json=payload)
        response.raise_for_status()

    generated_text = response.json()[0]['generated_text']

    # Split response into sentences
    sentences = split_sentences(generated_text)

    # Filter out sentences that are similar to the user_input
    filtered_sentences = [sentence for sentence in sentences if not is_similar_to_input(sentence, user_input)]

    # Join the filtered sentences and return the final response
    return ' '.join(filtered_sentences)


def main():
    st.set_page_config(page_title="Tony Stark and Harry Potter Chatbot", page_icon=None, layout='wide',
                       initial_sidebar_state='auto')

    character = st.sidebar.selectbox("Choose a character:", ("Tony Stark", "Harry Potter"))

    # Clear conversation history when switching characters
    session_state = st.session_state
    if "selected_character" not in session_state:
        session_state.selected_character = character
    else:
        if session_state.selected_character != character:
            session_state.selected_character = character
            if "conversation_history" in session_state:
                del session_state.conversation_history

    st.title("Tony Stark and Harry Potter Chatbot")
    st.markdown(f"<div style='background-color: rgba(136, 136, 136, 0.2); border-radius: 10px; padding: 10px; color: "
                f"white; margin-bottom: 20px;'> <b>System:</b> Now chatting with {character}</div>", unsafe_allow_html=True)

    if "conversation_history" not in session_state:
        session_state.conversation_history = []

    for message in session_state.conversation_history:
        if message["sender"] == "user":
            st.markdown(f"<div style='background-color: rgba(102, 102, 102, 0.2); border-radius: 10px; padding: 10px; "
                        f"color: white; margin-bottom: 10px;'><b>User:</b> {message['text']}</div>",
                        unsafe_allow_html=True)

        else:
            st.markdown(f"<div style='background-color: rgba(170, 170, 170, 0.2); border-radius: 10px; padding: 10px; "
                        f"color: white; margin-bottom: 10px;'>{message['text']}</div>", unsafe_allow_html=True)

    user_input = st.text_input("Type your message:", key="user_input")
    if st.button("Send"):
        if user_input.strip():
            session_state.conversation_history.append({"sender": "user", "text": user_input})

            if character == "Tony Stark":
                model_name = "diabolic6045/tony_stark_chatbot"
            elif character == "Harry Potter":
                model_name = "diabolic6045/harry_potter_chatbot"

            response = get_response(user_input, model_name)
            session_state.conversation_history.append({"sender": "character", "text": f"<b>{character}:</b> {response}"})
            st.experimental_rerun()


if __name__ == "__main__":
    main()
