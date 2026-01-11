from Bio import Entrez
from lxml import etree
from io import BytesIO
import re
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

Entrez.email = "charlie.kotula@gmail.com"
Entrez.api_key = "7cdd9ccffdb47c8d3b5e603a22f15ca54309"

# Search query for getting relevant research articles
query = """
(
  rehabilitation AND "physical therapy" OR "return to sport" OR "return to play"
) AND (
  injury OR surgery OR postoperative OR musculoskeletal
) AND (
  exercise OR "therapeutic exercise" OR training
) AND (
  review[pt] OR systematic review[pt] OR meta-analysis[pt]
)
"""

def get_ids_with_metadata(query):
    """
    Gets PMC article UIDs, PMCIDs, and titles based on search query

    Args: query (str) - search query used to retrieve PMC articles
    Returns: metadata (dict) - dictionary containing PMCID and titles corresponding
        to the articles UID
    """
    # Get relevant UIDs and titles
    metadata = []
    handle = Entrez.esearch(
        db='pmc',
        term=query,
        retmax=10, # CHANGE
    )
    
    # Get relevant articles
    uids = Entrez.read(handle)['IdList']
    handle.close()
    
    # Get summaries for metadata
    summary = Entrez.esummary(
        db='pmc',
        id=','.join(uids)
    )
    records = Entrez.read(summary)
    
    # Map UIDs to titles and pmids
    for rec in records:
        title = rec['Title'].lower()
        title = re.sub(r'[^a-z0-9]+', '_', title)
    
        metadata.append(
            {
                'uid': rec['Id'],
                'pmcid': rec['ArticleIds']['pmcid'],
                'title': title
            }
        )

    return metadata

######################################################
# Functions to extract, clean, and chunk text from PMC
def get_xml(pmc_id):
    """
    Returns the xml tree representation of the PMC article corresponding to
    the input UID.
    """
    handle = Entrez.efetch(
        db='pmc',
        id=pmc_id,
        retmode='xml',
        # rettype='full'
    )

    xml_dat = handle.read()

    # Converts xml bytes to tree
    xml_tree = etree.parse(BytesIO(xml_dat))
    
    return xml_tree

def get_text(xml):
    """
    Returns a dictionary of {section title: content} for the xml tree root.
    """
    text = []

    root = xml.getroot()

    # Remove references
    for xref in root.xpath('.//xref'):
        parent = xref.getparent()
        if parent is None:
            continue

        # removes punction surrounding references
        prev = xref.getprevious()

        # Handles punctuation before ref
        if prev is not None and prev.tail is not None:
            prev.tail = re.sub(r'[\[\(]\s*$', ' ', prev.tail)
        else:
            # xref is the first child → clean parent.text
            if parent.text:
                parent.text = re.sub(r'[\[\(]\s*$', ' ', parent.text)   

        # Handles punctuation after ref
        if xref.tail:
            xref.tail = re.sub(r'^\s*[\]\)]*', ' ', xref.tail)
            
        parent.remove(xref)
            
    
    for sec in root.xpath('.//body//sec'):
        title = sec.findtext('title')
        if not title:
            continue
        title = title.lower()

        # Gets paragraphs from each section
        paragraphs = [
            ''.join(p.itertext()) for p in sec.findall('p')
        ]
    
        # Add sections to sections list
        if paragraphs: # ignores empty sections
            text.append((title , ' '.join(paragraph for paragraph in paragraphs)))
    
    return text

def clean_text(text):
    """
    Cleans article text, removing extra spaces, etc.
    """
    cleaned_text = []
    for section, words in text:
        words = re.sub(r'\s+', ' ', words)
        words = words.replace('\xa0', ' ').strip()
        cleaned_text.append((section, words))
    
    return cleaned_text

def chunk_text(cleaned_text, uid, pmcid, title,):
    """
    Takes cleaned article text and chunks it into LangChain Documents
    """
    docs = []
    
    for section, text in cleaned_text:
        # Create section label for metadata
        section = section.lower()
        section = re.sub(r'[^a-z0-9]+', '_', section)
    
        # Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=['\n\n', '\n', '. ', ' ', '']
        )
        chunks = splitter.split_text(text)
    
        # Create langchain Documents
        for i, chunk in enumerate(chunks):
            doc=Document(
                page_content=chunk,
                metadata={
                    "uid": uid,
                    "pmcid": pmcid,
                    "article": title,
                    "section": section,
                    "chunk_id": f'{uid}-{section}-{i}'
                }
            )

            docs.append(doc)

    return docs

def process_article(article):
    # get metadata for Document creation
    uid = article['uid']
    pmcid = article['pmcid']
    title = article['title']
    
    # extract sections
    xml = get_xml(uid)
    text = get_text(xml)

    # clean sections
    cleaned_text = clean_text(text)

    # chunk text and create LangChain Documents with metadata
    docs = chunk_text(cleaned_text, uid, pmcid, title)
    return docs


if __name__ == "__main__": 
    # Create metadata list to be used in multiprocessing
    metadata = get_ids_with_metadata(query)
    print(f'retrieved {len(metadata)} articles')
    
    #### Multiprocessing of articles ####
    documents = []

    start_time = time.time()
    ### Processes multiple articles in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_article, article) for article in metadata]
    
        for future in as_completed(futures):
            documents.extend(future.result())

        end_time = time.time()
        print(f'processed {len(documents)} documents in {end_time - start_time} seconds')

