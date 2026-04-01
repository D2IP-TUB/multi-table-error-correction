import requests
from bs4 import BeautifulSoup

# Function to extract specific links from a webpage
def extract_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all 'a' tags with href attribute
    links = soup.find_all('a', href=True)

    # Filter links containing '7z' and 'pages-meta-history'
    filtered_links = [link['href'] for link in links if '.7z' in link['href'] and 'pages-meta-history' in link['href']]

    return filtered_links

# URL of the webpage to scrape
url = 'https://dumps.wikimedia.org/enwiki/20240201'

# Extract and print the filtered links
filtered_links = extract_links(url)

with open('wiki_revisions_urls.txt', 'w') as f:
    for link in filtered_links:
        f.write("https://dumps.wikimedia.org" + link + '\n')

        # https://dumps.wikimedia.org/enwiki/20240201/enwiki-20240201-pages-meta-history7.xml-p1533342p1561523.7z