import os
import random

def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = []
        current_sentence = []

        for line in file:
            if line.strip() == '':  # New sentence begins
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)

        # Append the last sentence if the file does not end with a blank line
        if current_sentence:
            sentences.append(current_sentence)
    return sentences


def write_sentences(sentences, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            for line in sentence:
                file.write(line)
            file.write('\n')  # Add a blank line between sentences


def sample_sentences(input_dir, output_file_path, sample_size=400):
    all_sampled_sentences = []

    # Loop through each dataset folder in 'Datasets'
    for dataset_name in os.listdir(input_dir):
        dataset_path = os.path.join(input_dir, dataset_name)
        if os.path.isdir(dataset_path) and dataset_name != 'NewData':
            file_path = os.path.join(dataset_path, output_file_path)
            sentences = read_sentences(file_path)

            if len(sentences) < sample_size:
                raise ValueError(f"Not enough sentences in {file_path} to sample {sample_size} sentences.")

            sampled_sentences = random.sample(sentences, sample_size)
            all_sampled_sentences.extend(sampled_sentences)

    # Write all sampled sentences to the new file
    write_sentences(all_sampled_sentences, os.path.join(input_dir, 'NewData', output_file_path))


def main():
    datasets_dir = 'Datasets'
    new_data_dir = os.path.join(datasets_dir, 'NewData')

    # Ensure the output directory exists
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)

    # Output files for the combined sampled data
    for split in ['train', 'test', 'valid']:
        output_file = f'{split}.txt'
        sample_sentences(datasets_dir, output_file, sample_size=400 if split == 'train' else 80)


if __name__ == '__main__':
    main()
