from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated, List, Any

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from dotenv import load_dotenv

from openai import AsyncOpenAI

from supabase import Client


from groq import AsyncGroq, Groq
import openai

import logfire
import os

# Import the message classes from Pydantic AI
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter
)

# ENV
from econ_researcher  import econ_researcher, EconomicResearchDeps, list_document_pages_helper



# Load environment variables
load_dotenv()

# Suppress logfire warnings (optional)
logfire.configure(send_to_logfire='never')

# API & LLMs Configuration
base_url = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")  # Default to Groq API
LLM_API_KEY = os.getenv("GROQ_API_KEY", "no-llm-api-key-provided")  # Groq API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI Key (for embeddings)


# Initialize OpenAI Client for LLM (Groq) and Embeddings (OpenAI)
openai_client = AsyncOpenAI(
    base_url=base_url,  # Use Groq for LLM requests
    api_key=LLM_API_KEY
)

# PITCHER AGENT
#load reasoner model from environment
REASONER_LLM_MODEL = os.getenv("REASONER_MODEL", "llama-3.1-8b-instant")
base_url = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")  # Default to Groq API
api_key = os.getenv("GROQ_API_KEY", "no-llm-api-key-provided")

#Innitialize Pitcher agent
pitcher = Agent(
    OpenAIModel(REASONER_LLM_MODEL, base_url=base_url, api_key=api_key),
    system_prompt="You are a reasoning AI agent that learn from the corpus of Marxian/classical economic books that we input here. Let's brainstorm some research ideas from to the query of users and guide the reasoning process efficiently."
)



# Load primary LLM model from environment (default to DeepSeek)
PRIMARY_LLM_MODEL = os.getenv("PRIMARY_MODEL", "deepseek-r1-distill-qwen-32b")

# Ensure the correct API & Base URL are used
BASE_URL = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")  # Default: Groq API
LLM_API_KEY = os.getenv("LLM_API_KEY", "no-llm-api-key-provided")  # Groq API Key



#ROUTING AGENT:  for routing
# Initialize Routing Agent
router_agent = Agent(
    OpenAIModel(PRIMARY_LLM_MODEL, base_url=base_url, api_key=api_key),
    system_prompt=(
    "Your job is to route user message either to the end of the conversation"
    "or to continue produce research proposal."
    ),
)

# ENDING AGENT  ==> Terminate the conversation
# Initialize the LLM model with correct environment variables
end_conversation_agent = Agent(
    OpenAIModel(PRIMARY_LLM_MODEL, base_url=base_url, api_key=api_key),
    system_prompt=(
        "Your job is to end the conversation for creating a package of research proposal"
        "Finish by instructing user how to use the proposal"
    ),
)

# Initialize Supabase Client
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Define AgentState

class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    pitch: str

# PITCHER with REASONER LLMs
	
async def pitching_research_ideas(state: AgentState): 
    # First, get the document pages so the reasoner can decide which ones are necessary
    document_pages = await list_document_pages_helper(supabase)
    document_pages_str = "\n".join(document_pages)

    # Then, use the reasoner to pitch ideas
    prompt = f"""
    User AI Agent Request: {state['latest_user_message']}
    
    Create or the AI agent including:
    - Architecture diagram
    - Core components
    - External dependencies
    - Testing strategy

    Also based on these document pages or texts available from the economic corpus:

    {document_pages_str}

    Include a list of document pages that are relevant to creating this agent for the user in the pitcher document.
    """

    result = await pitcher.run(prompt)
    pitch = result.data

    # Save the pitch to a file
    pitch_path = os.path.join("workbench", "pitch.md")
    os.makedirs("workbench", exist_ok=True)

    with open(pitch_path, "w", encoding="utf-8") as f:
        f.write(pitch)
    
    return {"pitch": pitch}

# RESEARCHING MATERIAL WITH FEEDBACK HANDLING	

async def researcher_proposal_agent(state: AgentState, writer):    
    # Prepare dependencies
    deps = EconomicResearchDeps(
        supabase=supabase,
        openai_client=openai_client,
        reasoner_output=state['pitch']
    )

    # Get the message history into the format
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent in a stream
    # if is_ollama:
    #     writer = get_stream_writer()
    #     result = await econ_researcher.run(state['latest_user_message'], deps=deps, message_history= message_history)
    #     writer(result.data)
    # else:
    async with econ_researcher.run_stream(
            state['latest_user_message'],
            deps=deps,
            message_history= message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    # print(ModelMessagesTypeAdapter.validate_json(result.new_messages_json()))

    return {"messages": [result.new_messages_json()]}


# Interrupt the graph to get the user's next message ==> HUMAN LOOP CRITIQUES
 

def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "latest_user_message": value
    }

# Determine if the user is finished creating their AI agent or not (PROMPT)

async def route_user_message(state: AgentState):
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation, respond with just the text "finish_conversation".
    If the user wants to continue adjust or critique the proposal, respond with just the text "researcher_proposal_agent".
    """

    result = await router_agent.run(prompt)
    next_action = result.data

    if next_action == "finish_conversation":
        return "finish_conversation"
    else:
        return "researcher_proposal_agent"

# End of conversation agent to give instructions for executing the agent

async def finish_conversation(state: AgentState, writer): 
   
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent in a stream
    # if is_ollama:
    #     writer = get_stream_writer()
    #     result = await end_conversation_agent.run(state['latest_user_message'], message_history= message_history)
    #     writer(result.data)   
    # else: 
    async with end_conversation_agent.run_stream(
            state['latest_user_message'],
            message_history= message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    return {"messages": [result.new_messages_json()]}




# Build workflow
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("pitching_research_ideas", pitching_research_ideas)
builder.add_node("researcher_proposal_agent", researcher_proposal_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("finish_conversation", finish_conversation)


# Set edges
builder.add_edge(START, "pitching_research_ideas")
builder.add_edge("pitching_research_ideas", "researcher_proposal_agent")
builder.add_edge("researcher_proposal_agent", "get_next_user_message")
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    {"researcher_proposal_agent": "researcher_proposal_agent", "finish_conversation": "finish_conversation"}
)
builder.add_edge("finish_conversation", END)


# Configure persistence
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)