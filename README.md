# BioBERT FineTuning Reproduction

## Overview

This project attempts to reproduce the fine-tuning of BioBERT v1.1 for the task of Named Entity Recognition. This involves applying BioBERT over a range of biomedical datasets, including original datasets used in the BioBERT paper and new datasets crafted for this project. The main purpose is to try to validate the results from the original BioBERT study and see if the same results can be achieved using newly crafted datasets.
## Project Structure

### Files and Directories
- **Datasets**: This folder contains two subfolders:
- **OriginalDatasets**: Includes the seven datasets originally used in the BioBERT paper for benchmarking NER capabilities. Each dataset is in BIO format and tagged with a single entity type.
- **NewDataSets**: Includes five newly created datasets using different methods.
- **GeneratedData**: A dataset with 30,000 sentences, generated using a set of 200 entity-free sentences and 200 sentences with random entities. This is meant for experimental purposes only.
- **Compiled-Disease**: Integrates BC5CDR-Disease and NCBI-Disease datasets.
- **Compiled-Multi**: Compiled a portion of all the raw datasets together and included all the available entity tags.
- **Scraped-Disease**: Contains roughly half of the scraped epigenetic abstracts annotated with entity label "Disease".
- **Scraped-Multi**: The 1,500 scraped epigenetic abstracts formatted to BIO and annotated with Chemical, Species, Gene and Disease entities.

- **NewDataScrapeGenerator**: Scripts for generating new datasets.
- **NewDataScrape.py**: Scrapes abstracts of PubMed articles based on specified search terms.
- **NewDataOpenAPIAutoLabel.py**: Uses the OpenAI API to auto-label entities in the scraped abstracts.
- BioBERTDatasetFinetuning.py: Script to fine-tune BioBERT on NER datasets.

### Development Environment
-.devcontainer: Configures a development container to run the project in a reproducible environment using VSCode. The container installs necessary Python libraries and uses Docker to ensure consistency.
- **Dockerfile**: It sets up the development container with Python 3.9, the necessary dependencies such as PyTorch, and all other required libraries.
- **requirements.txt**: Lists all Python dependencies required for the project.

## Installation Instructions

### Prerequisites
- Install Docker to use provided dev container.
Clone this repository to your local machine.

### Running the Development Container
1. Open the repository in VSCode.
2. When prompted, re-open the folder in the container to use the `.devcontainer` configuration.
3. The container will automatically install all the required dependencies specified in `requirements.txt`. To use the web scraper and autolabeler, also install `bs4` and `openai` using:
```sh
pip install beautifulsoup4 openai
```

### Data Preparation
- **Raw Datasets**: No additional preprocessing is needed. All datasets are in BIO format for NER tasks.
- **New Datasets**: To create new datasets:
1. **Scrape Abstracts**: In `NewDataScrape.py`, change the search term and number of articles (e.g., line 46: `abstracts = scrape_pubmed("epigenetics", num_articles=1500)`). Then run the script.
2. **Autolabel Data**: After scraping, put the file path to the output in `NewDataOpenAPIAutoLabel.py`. Run the script to auto-label entities. Note that OpenAI API key is required.
3. Preprocess Data: A little bit more manual preprocessing is required beyond the auto-labeling, including format correction, labeling supervision, and dataset division into `train.txt`, `test.txt,` and `valid.txt`.

### Fine-Tuning BioBERT
To fine-tune BioBERT:
1. Set the `DATASET_NAME` variable in `BioBERTDatasetFinetuning.py` to the desired dataset directory under `Datasets/`.
2. Ensure that the dataset directory has `train.txt`, `test.txt`, and `valid.txt` files in BIO style for NER.
3. Run the script to start the fine-tuning process.

## Dataset Details
- **Original Datasets**: These datasets are annotated with only one entity per sentence in BIO format. Here is an example:

```
block NN O
after IN O
previous JJ O
sections NNS O
.. O

Chloramphenicol NNP B-Gene
acetyltransferase NN I-Gene
assays VBZ O
examining VBG O
```

- **New Datasets**:

- **GeneratedData**: Artificial data with a balanced set of sentences containing and not containing entities.
- **Compiled-Disease**: Joined disease-related datasets for benchmarking.
- **Compiled-Multi**: A combination of several entity types.
- **Scraped-Multi** and **Scraped-Disease**: Built from PubMed abstracts with annotations for multiple or single entity types, respectively.

## Run the Project

- **Scraping and Autolabeling**:

1. Run `NewDataScrape.py` to scrape abstracts.

2. Pipe the output to script `NewDataOpenAPIAutoLabel.py` for abstract labeling. 

3. Perform preprocessing manually before feeding to the data to train. 

- **Fine-Tuning BioBERT**: 

Use `BioBERTDatasetFinetuning.py` with a structured dataset to fine-tune BioBERT for NER tasks. 

## Limitations and Future Work 

- The **GeneratedData** dataset is not suited for practical application as the generation technique is simplistic and does not have realistic context associated with it. 
- No plans to extend the project further, as the goal was focussed on reproducing results and testing new datasets. 

## References - BioBERT is realized via the **Hugging Face Transformers** library. 

- Got the original datasets from the GitHub page of **BioBBC**, which did a similar study. 

## Contributors 

- Written by Jack.