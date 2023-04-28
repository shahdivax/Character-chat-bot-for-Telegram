import logging
import re
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Import the fine-tuned model and tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("tony_stark_chatbot")
tokenizer = GPT2Tokenizer.from_pretrained("tony_stark_chatbot")

def split_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

def is_similar_to_input(sentence, user_input):
    return user_input.lower() in sentence.lower()

def get_response(user_input, temperature=0.8):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, temperature=temperature)
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

def start(update, context):
    update.message.reply_text("Hi! I am Tony Stark's chatbot. How can I help you?")

def chat(update, context):
    user_input = update.message.text
    response = get_response(user_input)
    update.message.reply_text(response)

def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    # Register handlers
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, chat))

    # Start the bot
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed or the process receives SIGINT, SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()