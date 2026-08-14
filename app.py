import streamlit as st
import json
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

# Import our tools directly from the ag_server module
from ag_server import get_current_weather, calculate_claim

# Load environment variables from the hidden .env file
load_dotenv()

# -------------------------------------------------------------
# 1. UI Setup & Initialization
# -------------------------------------------------------------
st.title("🌱 Climate Risk & Ag-Insurance Agent")
st.caption("Phase 4: Modular Architecture (Secure Environment Variables)")

# Initialize Groq using the secure environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API Key not found! Please set it in your .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Define the Groq tool schemas manually mapping to our imported functions
groq_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current real-time weather (temperature and precipitation) for a specific latitude and longitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "The latitude of the location."},
                    "longitude": {"type": "number", "description": "The longitude of the location."}
                },
                "required": ["latitude", "longitude"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_claim",
            "description": "Calculates the exact crop insurance claim payout based on Threshold Yield, Actual Yield, and Sum Insured.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold_yield": {"type": "number", "description": "The historical average threshold yield in kg/hectare."},
                    "actual_yield": {"type": "number", "description": "The actual harvested yield in kg/hectare."},
                    "sum_insured": {"type": "number", "description": "The total financial sum insured."}
                },
                "required": ["threshold_yield", "actual_yield", "sum_insured"]
            }
        }
    }
]

# -------------------------------------------------------------
# 2. Connect to Database & Load Memory
# -------------------------------------------------------------
@st.cache_resource
def load_vector_db():
    embeddings = FastEmbedEmbeddings()
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an Ag-Insurance expert. Use context for policy rules, and call tools for weather or math calculations."}
    ]

for message in st.session_state.messages:
    if message["role"] not in ["system", "tool"]:
        if not (message["role"] == "assistant" and message.get("tool_calls")):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# -------------------------------------------------------------
# 3. The Agent Loop
# -------------------------------------------------------------
if prompt := st.chat_input("Ask about policies, check live weather, or calculate a claim..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Perform RAG Search for policy context
    results = vector_db.similarity_search(prompt, k=10)
    context = "\n\n".join([doc.page_content for doc in results])

    # Build a clean list of messages for Groq, stripping out any unsupported fields
    clean_messages = []
    for m in st.session_state.messages:
        if m["role"] == "system":
            clean_messages.append({"role": "system", "content": m["content"]})
        elif m["role"] == "user":
            clean_messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg_dict = {"role": "assistant", "content": m.get("content")}
            if m.get("tool_calls"):
                msg_dict["tool_calls"] = m["tool_calls"]
            clean_messages.append(msg_dict)
        elif m["role"] == "tool":
            clean_messages.append({
                "role": "tool",
                "tool_call_id": m["tool_call_id"],
                "name": m["name"],
                "content": m["content"]
            })

    # Inject the RAG context into the system prompt
    clean_messages[0] = {
        "role": "system",
        "content": f"""You are an Ag-Insurance expert. Use policy context below for rules. Use your tools for weather or math calculations.
        Policy Context: {context}"""
    }

    # Ask Groq to decide what to do
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=clean_messages,
        tools=groq_tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message

    # Execute tools locally if requested
    if response_message.tool_calls:
        # Save assistant's tool call request to session state safely
        assistant_msg_dict = {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [tc.model_dump() for tc in response_message.tool_calls]
        }
        st.session_state.messages.append(assistant_msg_dict)
        clean_messages.append(assistant_msg_dict)
        
        for tool_call in response_message.tool_calls:
            t_name = tool_call.function.name
            t_args = json.loads(tool_call.function.arguments)
            
            st.info(f"🔌 Executing server tool -> `{t_name}` with args: {t_args}")
            
            if t_name == "get_current_weather":
                tool_output = get_current_weather(t_args["latitude"], t_args["longitude"])
            elif t_name == "calculate_claim":
                tool_output = calculate_claim(t_args["threshold_yield"], t_args["actual_yield"], t_args["sum_insured"])
            else:
                tool_output = json.dumps({"error": "Tool not found"})
            
            tool_response_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": t_name,
                "content": tool_output
            }
            st.session_state.messages.append(tool_response_msg)
            clean_messages.append(tool_response_msg)

        # Final summary back from Groq
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=clean_messages
        )
        
        final_text = final_response.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})
        
    else:
        final_text = response_message.content
        with st.chat_message("assistant"):
            st.markdown(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})