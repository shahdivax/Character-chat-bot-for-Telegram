import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("harry_potter_chatbot") #tony_stark_chatbot
tokenizer = GPT2Tokenizer.from_pretrained("harry_potter_chatbot")


def get_response(user_input, temperature=0.9):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2,
                            temperature=temperature)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Chatbot loop
while True:
    # Get user input
    user_input = input("You: ")

    # Check if the user wants to exit
    if user_input.lower() in ["exit", "quit"]:
        break

    # Generate response
    response = get_response(user_input, temperature=0.9)

    # Display response
    print(response)
