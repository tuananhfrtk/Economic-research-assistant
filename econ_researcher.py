from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List
from sentence_transformers import SentenceTransformer
from groq import AsyncGroq, Groq

# ENV

# Load environment variables
load_dotenv()

# Use DeepSeek as the default model
llm = os.getenv("PRIMARY_MODEL", "llama-3.1-8b-instant")

# Ensure BASE_URL points to Groq API unless otherwise specified
base_url = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")

# Correct API key assignment
api_key = os.getenv("GROQ_API_KEY", "no-llm-api-key-provided")

# Initialize LLM model
model = OpenAIModel(llm, base_url=base_url, api_key=api_key)

# Configure Logfire
logfire.configure(send_to_logfire="if-token-present")

embedding_model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')  # Hugging Face Embeddings
# Initialize Hugging Face embedding model
hf_model = SentenceTransformer(embedding_model_name)


# 1. Class of Agentic Teams
@dataclass
class EconomicResearchDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    reasoner_output: str

# 2. System Prompt 
system_prompt = """
[Role & Context]
You are an expert in classical and Marxian economic theories, specializing in constructing rigorous research plans and proposals. 
You have comprehensive access to the Marxian economic texts from marxists.org and must use them as primary references for scholarly insights.

[Core Responsibilities]
Your goal is to systematically develop a robust research proposal by addressing:
Background – Establish the context and significance of the study.
Objectives – Define clear and precise research goals.
Research Questions/Hypotheses – Formulate key questions or hypotheses guiding the study.
Scope – Clearly define the boundaries of the research (inclusions/exclusions).
Literature Review – Summarize existing research, identify gaps, and justify the study's relevance.

[Interaction Guidelines]
Take immediate action without requesting permission.
Verify sources and documents before drawing conclusions.
Provide honest, constructive feedback on content gaps.
Offer specific enhancement suggestions to improve clarity and coherence.
Encourage user feedback on proposed research frameworks.
Ensure consistency and logical coherence in structured outputs.

[Research Methodology & Reasoning]
Apply dialectical reasoning to analyze contradictions in economic theories.
Justify claims using direct citations from the Marxian corpus.
Use structured formatting (e.g., bullet points, paragraphs, or academic-style writing).
Encourage iterative refinement by requesting user feedback on drafts.
"""

# 3. Economic Researcher
econ_researcher = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=EconomicResearchDeps,
    retries=2
)
# 4. INSTRUCTION FROM LLM
@econ_researcher.system_prompt  
def add_reasoner_output(ctx: RunContext[str]) -> str:
    return f"""
    \n\nAdditional thoughts/instructions from the reasoner LLM. 
    This scope includes texts for you to search as well: 
    {ctx.deps.reasoner_output}
    """

# 1. GET EMBEDDINGS VECTOR FROM HUGGING FACE

# async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
#     """Get embedding vector from OpenAI."""
#     try:
#         response = await openai_client.embeddings.create(
#             model="text-embedding-3-small",
#             input=text
#         )
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"Error getting embedding: {e}")
#         return [0.0] * 1536  # Return zero vector on error
    
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector using Hugging Face's sentence-transformers."""
    try:
        # Generate embedding using Hugging Face model    
        response = hf_model.encode(text, convert_to_numpy=True).tolist()
        return response
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return [0.0] * hf_model.get_sentence_embedding_dimension()  # Return a zero vector of correct dimension on failure



    
# 2. Retrieve relevant document chunks based on the query with RAG.

@econ_researcher.tool
async def retrieve_relevant_document(ctx: RunContext[EconomicResearchDeps], user_query: str) -> str:
    """
    Retrieve relevant document chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant document chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': '{"source": "economic_assistant_docs"}'  # Ensure filter is a JSON string
            }
        ).execute()
        
        if not result.data:
            return "No relevant document found."
            
        # Format the results
        formatted_chunks = [
            f"# {doc['title']}\n\n{doc['content']}" for doc in result.data
        ]
        
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving document: {e}")
        return f"Error retrieving document: {str(e)}"
    

# 3. retrieve a list of all available archive pages.

async def list_document_pages_helper(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available economic document pages.
    This is called by the list_document_pages tool and also externally to fetch document pages for the reasoner LLM.
    
    Returns:
        List[str]: List of unique URLs for all document pages
    """
    try:
        # Query Supabase for unique URLs where source is economic_assistant_docs
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'economic_assistant_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        return sorted(set(doc['url'] for doc in result.data))
        
    except Exception as e:
        print(f"Error retrieving document pages: {e}")
        return [] 
    
#  4. Retrieve a list of all available Economics document pages.

@econ_researcher.tool
async def list_document_pages(ctx: RunContext[EconomicResearchDeps]) -> List[str]:
    """
    Retrieve a list of all available economic document pages.
    
    Returns:
        List[str]: List of unique URLs for all document pages
    """
    return await list_document_pages_helper(ctx.deps.supabase)


# 5. Retrieve the full content of a specific document page by combining all its chunks.

@econ_researcher.tool
async def get_page_content(ctx: RunContext[EconomicResearchDeps], url: str) -> str:
    """
    Retrieve the full content of a specific document page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'economic_assistant_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"] + [
            chunk['content'] for chunk in result.data
        ]
        
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

