# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
import os
import sys
import json
import asyncio
import requests
import openai
from xml.etree import ElementTree
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from urllib.parse import urljoin
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq, Groq




load_dotenv()

# Initialize OpenAI and Supabase clients

openai_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",  # Groq's endpoint
    api_key=os.getenv("GROQ_API_KEY")  # Ensure this is set in your .env
)


embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')  # Hugging Face Embeddings
# Initialize Hugging Face embedding model
hf_model = SentenceTransformer(embedding_model_name)

# Supabase client setup
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
# Verify Supabase connection
try:
    response = supabase.table('site_pages').select("*").limit(1).execute()
    if hasattr(response, 'error') and response.error:
        print(f"Supabase connection error: {response.error}")
        sys.exit(1)
    print("✅ Successfully connected to Supabase")
except Exception as e:
    print(f"❌ Failed to connect to Supabase: {e}")
    sys.exit(1)

# CLASS OF PROCESSED CHUNKS 
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

# 1. Split text into chunks, respecting code blocks and paragraphs.
def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks



# 2. Get title and summary
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("PRIMARY_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}




# 3. Get embedding vectors


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using Hugging Face's sentence-transformers."""
    try:
        # Generate embedding using Hugging Face model
        embedding = hf_model.encode(text, convert_to_numpy=True).tolist()
        return embedding
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return [0.0] * hf_model.get_sentence_embedding_dimension()  # Return a zero vector of correct dimension on failure


# 4. Process single chunk
async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "economic_assistant_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

# 5.  INsert chunks to supabase
async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        print(f"Attempting to insert data for {chunk.url} chunk {chunk.chunk_number}")
        result = supabase.table("site_pages").insert(data).execute()
        
        if hasattr(result, 'error') and result.error:
            print(f"Supabase error: {result.error}")
            return None
            
        print(f"✅ Successfully inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"❌ Error inserting chunk: {str(e)}")
        return None
    
# 6. Process a document and store its chunks in parallel.
async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)






def normalize_url(url: str) -> str:
    """
    Normalize URLs by removing fragments (#section) to avoid duplicate crawling of the same page.
    """
    parsed_url = urlparse(url)
    return parsed_url._replace(fragment="").geturl()

async def crawl_sequential(urls: List[str], depth: int = 4, current_depth: int = 1, visited: Set[str] = None):
    if visited is None:
        visited = set()

    if current_depth > depth:
        return

    print(f"\n=== Sequential Crawling (Depth {current_depth}) ===")

    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        session_id = "session1"
        new_urls = set()
        for url in urls:
            normalized_url = normalize_url(url)
            if normalized_url in visited:
                continue

            visited.add(normalized_url)
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )
            if result.success:
                print(f"Successfully crawled: {url}")
                print(f"Markdown length: {len(result.markdown_v2.raw_markdown)}")
                
                # PROCESS THE DOCUMENT HERE
                await process_and_store_document(url, result.markdown_v2.raw_markdown)
                
                # Extract new links if depth allows
                if current_depth < depth:
                    new_links = extract_links(url)
                    new_urls.update(new_links)
            else:
                print(f"Failed: {url} - Error: {result.error_message}")
        
        # Recursively crawl new URLs
        await crawl_sequential(list(new_urls), depth, current_depth + 1, visited)
    finally:
        await crawler.close()

def extract_links(base_url: str):
    """
    Extracts filtered links from a given page, excluding those inside elements with class 'title', 'footer', or 'information'.
    """
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        found_urls = set()
        
        # Remove unwanted sections before extracting links, but keep 'index'
        for unwanted_section in soup.find_all(class_=["title", "footer", "information"]):
            unwanted_section.decompose()
        
        for link in soup.find_all("a", href=True):
            # Ensure the link is not inside an unwanted section
            parent_classes = set(
                cls for parent in link.find_parents() if parent.has_attr("class") for cls in parent["class"]
            )
            
            if parent_classes.intersection({"title", "footer", "information"}):
                continue
            
            full_url = urljoin(base_url, link["href"])
            parsed_url = urlparse(full_url)
            normalized_url = normalize_url(full_url)
            
            # Allow .htm links with or without fragments (e.g., index.htm and index.htm#economy), but avoid duplicates
            if (
                any(keyword in full_url.lower() for keyword in ["note", "people"]) or
                full_url.endswith(".css") or
                (not parsed_url.path.endswith(".htm"))
            ):
                continue
            
            found_urls.add(normalized_url)
        
        return list(found_urls)
    except Exception as e:
        print(f"Error extracting links from {base_url}: {e}")
        return []

async def main():
    start_url = "https://www.marxists.org/archive/marx/works/1885-c2/index.htm"
    urls = extract_links(start_url)
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_sequential(urls, depth=1)
    else:
        print("No URLs found to crawl")

if __name__ == "__main__":
    asyncio.run(main())

