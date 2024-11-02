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


def sample_sentences(input_dir, output_dir, sample_size_train=400, sample_size_test_valid=80):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ['train', 'test', 'valid']:
        input_file_path = os.path.join(input_dir, f'{split}.txt')
        output_file_path = os.path.join(output_dir, f'{split}.txt')

        sentences = read_sentences(input_file_path)
        sample_size = sample_size_train if split == 'train' else sample_size_test_valid

        if len(sentences) < sample_size:
            raise ValueError(f"Not enough sentences in {input_file_path} to sample {sample_size} sentences.")

        sampled_sentences = random.sample(sentences, sample_size)
        write_sentences(sampled_sentences, output_file_path)


def main():
    datasets_dir = 'Datasets'
    new_data_dir = os.path.join(datasets_dir, 'NewData')

    # Loop through each dataset folder in 'Datasets'
    for dataset_name in os.listdir(datasets_dir):
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if os.path.isdir(dataset_path) and dataset_name != 'NewData':
            output_path = os.path.join(new_data_dir, dataset_name)
            sample_sentences(dataset_path, output_path)


if __name__ == '__main__':
    main()
