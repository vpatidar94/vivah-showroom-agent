import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate
import uuid

# Page config
st.set_page_config(
    page_title="VivahGPT Chatbot",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

@st.cache_resource
def initialize_agent():
    """Initialize the LangGraph agent (cached to avoid re-initialization)"""
    try:
        # LLM from Groq
        llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=os.getenv("GROQ_API_KEY")
        )

        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # MCP tool client
        client = MultiServerMCPClient(
            {
                "vivah_showroom_mcp": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                }
            }
        )

        # Get tools asynchronously
        tools = loop.run_until_complete(client.get_tools())
        
        # ReAct-style system prompt
        prefix = """
            You are VivahGPT, a helpful assistant managing showroom bookings.
            Use the following tools to answer user queries.
        """

        # Memory for state tracking
        memory = MemorySaver()

        # Create ReAct agent
        agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=memory
        )

        return agent, loop
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None, None

async def get_agent_response(agent, user_input, thread_id):
    """Get response from the agent"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        user_message = {
            "messages": [
                {"role": "user", "content": user_input}
            ]
        }

        response = await agent.ainvoke(user_message, config=config)

        if "messages" in response and response["messages"]:
            return response["messages"][-1].content
        else:
            return "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Title and description
    st.title("💍 VivahGPT Chatbot")
    st.markdown("Welcome to VivahGPT! Your AI assistant for showroom booking management.")

    # Sidebar
    with st.sidebar:
        st.header("Chat Settings")
        
        # Session management
        if st.button("🔄 New Chat Session"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        # Display current thread ID
        if 'thread_id' not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        
        st.info(f"Thread ID: {st.session_state.thread_id[:8]}...")
        
        # API status
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.success("✅ GROQ API Key loaded")
        else:
            st.error("❌ GROQ API Key not found")
            st.warning("Please set GROQ_API_KEY in your .env file")

        # Sample queries
        st.header("💡 Sample Queries")
        sample_queries = [
            "Which booking has the highest amount?",
            "Show me all bookings for today",
            "What are the available time slots?",
            "Cancel booking ID 123",
            "Create a new booking"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query[:20]}"):
                st.session_state.current_query = query

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm VivahGPT, your showroom booking assistant. How can I help you today?"
        })

    # Initialize agent
    agent, loop = initialize_agent()
    
    if agent is None:
        st.error("⚠️ Could not initialize the agent. Please check your configuration.")
        st.stop()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Handle sample query selection
    if hasattr(st.session_state, 'current_query'):
        user_input = st.session_state.current_query
        delattr(st.session_state, 'current_query')
    else:
        # Chat input
        user_input = st.chat_input("Type your message here...")

    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("VivahGPT is thinking..."):
                try:
                    # Run async function in the existing event loop
                    response = loop.run_until_complete(
                        get_agent_response(agent, user_input, st.session_state.thread_id)
                    )
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "VivahGPT - Powered by LangGraph & Groq LLaMA 4"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()