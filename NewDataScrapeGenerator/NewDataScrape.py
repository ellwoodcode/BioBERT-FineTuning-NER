import requests
from bs4 import BeautifulSoup
import time

# Function to scrape PubMed abstracts
def scrape_pubmed(keyword, num_articles=10, output_file='abstracts.txt'):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/?term="
    page = 2
    articles = []
    count = 0
    
    while count < num_articles:
        search_url = f"{base_url}{keyword}&page={page}"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find article links
        for link in soup.find_all('a', class_='docsum-title'):
            if count >= num_articles:
                break
            article_url = "https://pubmed.ncbi.nlm.nih.gov" + link.get('href')
            article_response = requests.get(article_url)
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            
            # Extract abstract text
            abstract = article_soup.find('div', class_='abstract-content')
            if abstract:
                articles.append(abstract.text.strip())
                print(f"Found Abstract {count}")
                count += 1
                if count >= num_articles:
                    break
        
        # Move to the next page
        page += 1
        time.sleep(1)  # To avoid overwhelming the server
    
    # Save abstracts to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, abstract in enumerate(articles):
            f.write(f"Abstract {i+1}:{abstract}")
    
    return articles

# Usage
abstracts = scrape_pubmed("epigenetics", num_articles=1500)
for i, abstract in enumerate(abstracts):
    print(f"Abstract {i+1}:", abstract)