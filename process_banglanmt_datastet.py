from datasets import load_dataset
import random

# Load the dataset
dataset = load_dataset("csebuetnlp/BanglaNMT", split="train") 

# File to write the sentences
output_file = "banglaNMT.txt"

# Function to write the dataset to a file
def write_dataset_to_file(dataset, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            bangla_sentence = row['bn']
            english_sentence = row['en']
            if random.random() > 0.5:
                line = f"Bangla: {bangla_sentence} English: {english_sentence}\n"
            else:
                line = f"English: {english_sentence} Bangla: {bangla_sentence}\n"
            f.write(line)

# Write the dataset to the file
write_dataset_to_file(dataset, output_file)

print(f"Data has been written to {output_file}")