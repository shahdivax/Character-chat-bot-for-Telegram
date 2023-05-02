import logging
import re
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Import the fine-tuned models and tokenizers
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the models and tokenizers
tony_stark_model = GPT2LMHeadModel.from_pretrained("tony_stark_chatbot")
tony_stark_tokenizer = GPT2Tokenizer.from_pretrained("tony_stark_chatbot")

harry_potter_model = GPT2LMHeadModel.from_pretrained("harry_potter_chatbot")
harry_potter_tokenizer = GPT2Tokenizer.from_pretrained("harry_potter_chatbot")


def split_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)


def is_similar_to_input(sentence, user_input):
    return user_input.lower() in sentence.lower()


def get_response(user_input, model, tokenizer, temperature=0.9, top_p=0.9):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # Create an attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).float()

    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2,
                            temperature=temperature, attention_mask=attention_mask, do_sample=True,
                            top_p=top_p)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Split response into sentences
    sentences = split_sentences(response)

    # Filter out sentences that are similar to the user_input
    filtered_sentences = [sentence for sentence in sentences if not is_similar_to_input(sentence, user_input)]

    # Join the filtered sentences and return the final response
    return ' '.join(filtered_sentences)


# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace YOUR_TOKEN with the token you received from the BotFather
TELEGRAM_TOKEN = "6097706211:AAGBo2_qECiFAc0DydNq89QKUmV1fELqkxY"


def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hi! I am Tony Stark and Harry Potter's chatbot.")
    update.message.reply_text("Please use /tony_stark or /harry_potter to chat with the respective characters.")


def chat_tony_stark(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    context.user_data[chat_id] = 'tony_stark'
    update.message.reply_text("Chatting with Tony Stark.")


def chat_harry_potter(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    context.user_data[chat_id] = 'harry_potter'
    update.message.reply_text("Chatting with Harry Potter.")


def chat(update: Update, context: CallbackContext):
    user_input = update.message.text
    chat_id = update.message.chat_id

    if chat_id not in context.user_data:
        update.message.reply_text("Please use /tony_stark or /harry_potter to chat with the respective characters.")
        return

    character = context.user_data[chat_id]

    if character == 'tony_stark':
        response = get_response(user_input, tony_stark_model, tony_stark_tokenizer)
    elif character == 'harry_potter':
        response = get_response(user_input, harry_potter_model, harry_potter_tokenizer)
    else:
        update.message.reply_text("Please use /tony_stark or /harry_potter to chat with the respective characters.")
        return

    update.message.reply_text(response)

def restart(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    if chat_id in context.user_data:
        del context.user_data[chat_id]
    update.message.reply_text("Restarting the conversation. Please use /tony_stark or /harry_potter to chat with the "
                              "respective characters.")


def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    # Register handlers
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("tony_stark", chat_tony_stark))
    dp.add_handler(CommandHandler("harry_potter", chat_harry_potter))
    dp.add_handler(CommandHandler("restart", restart))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, chat))

    # Start the bot
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()


if __name__ == '__main__':
    main()
