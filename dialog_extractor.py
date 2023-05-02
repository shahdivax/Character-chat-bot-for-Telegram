import re


# Note you should run this file only for tony stark chatbot

def parse_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def is_tony_stark_dialogue(line):
    tony_markers = [r"Tony Stark:", r"Tony:", r"TONY STARK (2012):", r"IRON MAN:", r"Iron Man:"]
    line_lower = line.lower()
    return any(re.match(marker.lower(), line_lower) for marker in tony_markers)


def clean_dialogue(line):
    line = re.sub(r'\(.*\)', '', line)  # Remove text in parentheses
    line = re.sub(r'\[.*\]', '', line)  # Remove text in brackets
    line = re.sub(r'<.*>', '', line)  # Remove text in angle brackets
    line = re.sub(r'[\n\t\r]+', ' ', line)  # Replace newline, tab, and carriage return characters with a space
    line = ' '.join(line.split())  # Remove extra spaces
    return line.strip()


def extract_tony_stark_dialogues(txt_files):
    tony_dialogues = []
    for txt_file in txt_files:
        lines = parse_txt(txt_file)
        for line in lines:
            if is_tony_stark_dialogue(line):
                cleaned_line = clean_dialogue(line)
                if cleaned_line:
                    tony_dialogues.append(cleaned_line)
    return tony_dialogues


# List of TXT files
txt_files = ['IronMan1.txt', 'IronMan2.txt', 'IronMan3.txt', 'Avengers.txt', 'AvengersEG.txt',
             'AvengersIW.txt', 'CivilWar.txt']

# Extract Tony Stark's dialogues
tony_dialogues = extract_tony_stark_dialogues(txt_files)

# Save the Tony Stark's dialogues to a merged file
with open('tony_stark_dialogues.txt', 'w', encoding='utf-8') as f:
    for dialogue in tony_dialogues:
        f.write(dialogue + '\n')
    print('tony_stark_dialogues.txt created')
