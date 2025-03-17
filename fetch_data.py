import requests
from xml.etree import ElementTree

def fetch_arxiv_papers(_, query="deep learning", max_results=100):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    root = ElementTree.fromstring(response.content)
    
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        pdf_link = entry.find("{http://www.w3.org/2005/Atom}link[@title='pdf']")
        pdf_url = pdf_link.attrib["href"] if pdf_link is not None else ""
        papers.append({"title": title, "summary": summary, "pdf_url": pdf_url})

    return papers

def download_pdf(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)