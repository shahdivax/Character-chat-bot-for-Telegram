import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datasets import Dataset


def read_dialogues_from_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        dialogues = f.readlines()
    return [d.strip() for d in dialogues]


def create_datasets(dialogues, tokenizer, test_size=0.1):
    train_dialogues, val_dialogues = train_test_split(dialogues, test_size=test_size)

    tokenized_train_dialogues = tokenizer(train_dialogues, return_tensors='pt', truncation=True, padding="max_length",
                                          max_length=50)
    tokenized_val_dialogues = tokenizer(val_dialogues, return_tensors='pt', truncation=True, padding="max_length",
                                        max_length=50)

    train_dataset = Dataset.from_dict(tokenized_train_dialogues)
    val_dataset = Dataset.from_dict(tokenized_val_dialogues)

    return train_dataset, val_dataset


# Load DialoGPT model and tokenizer
model = GPT2LMHeadModel.from_pretrained(
    "microsoft/DialoGPT-medium")  # use "microsoft/DialoGPT-large" for more creativity
tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")  # "microsoft/DialoGPT-large"

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Read Tony Stark's dialogues from a text file
dialogues_file = "harry_rows_final.txt"  # "tony_stark_dialogues.txt" for tony stark
tony_dialogues = read_dialogues_from_file(dialogues_file)

# Create train and validation datasets
train_dataset, val_dataset = create_datasets(tony_dialogues, tokenizer)

# Define training arguments and create a Trainer
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_steps=100,
    save_steps=100,
    warmup_steps=100,
    learning_rate=5e-5,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the DialoGPT model on Tony Stark's dialogues
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("harry_potter_chatbot")  # use "tony_stark_chatbot" for tony stark
tokenizer.save_pretrained("harry_potter_chatbot")

# Load the fine-tuned model and use it for generating responses
model = GPT2LMHeadModel.from_pretrained("harry_potter_chatbot")  # use "tony_stark_chatbot" for tony stark
tokenizer = GPT2Tokenizer.from_pretrained("harry_potter_chatbot")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_response(user_input, temperature=0.8):
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2,
                            temperature=temperature)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Example: Get a response for a user input
user_input = "Are you harry potter?" # "Are you tony stark?" for tony stark
response = get_response(user_input, temperature=0.8)
print("\nresponse : "+response)
