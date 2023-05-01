# Character Chat Bot for Telegram

This repository contains a chat bot designed for Telegram that can chat like famous movie characters such as Tony Stark (Iron Man) and Harry Potter. Engage in fun conversations with your favorite characters and see how they would respond to you.

## Getting Started

To access the code and set up the chat bot, make sure to use the `master` branch of this repository:

1. Clone the `master` branch of this repository:
```
git clone https://github.com/shahdivax/Character-chat-bot-for-Telegram.git --branch master
```

Then, follow the previously mentioned steps to set up and run the chat bot.

## Branch Information

All the code for this project is available in the `master` branch. This branch contains the latest updates and improvements to the chat bot. If you encounter any issues or would like to contribute, please submit a pull request or report an issue on the `master` branch.

2. Change into the project directory:
```
cd Character-chat-bot-for-Telegram
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run `dialog_extractor.py` to extract the dialogues from movie scripts:
```
python dialog_extractor.py
```

5. Run `modelcreation.py` to create the chat bot model:
```
python modelcreation.py
```

6. Run `bot.py` to run the bot locally, or run `TelegramBotRunner_of_Tonystark.py` to start the bot on Telegram:
```
python bot.py
```
or
```
python TelegramBotRunner_of_Tonystark.py
```

7. Access the Telegram bot on [t.me/DJS_Movie_Characters_Bot](https://t.me/DJS_Movie_Characters_Bot) after running `TelegramBotRunner_of_Tonystark.py`.

## Features

- Realistic conversations with your favorite movie characters
- Easy setup and deployment on Telegram
- Extendable to other movie characters by modifying the dataset
- Fun and engaging way to interact with friends and fans of the movies

## Contributing

Feel free to contribute to this project by submitting a pull request or reporting any issues you encounter. We appreciate your help in making this chat bot even better!

## Model

This chat bot is powered by the [DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) model, which is a powerful language model created by Microsoft. We fine-tuned DialoGPT-medium specifically for our movie characters to ensure realistic and engaging conversations. By using this advanced model, our chat bot can generate more accurate and context-aware responses, making the conversations feel like you're really talking to Tony Stark or Harry Potter.

## Acknowledgements

- Thanks to Microsoft for creating the DialoGPT-medium model
- Thanks to the creators of the movie scripts used in this project
- Special thanks to the developers of the libraries and tools used in building this chat bot
