# ==========================================================
# ⚖️ Nyaya Setu — Unified Streamlit + MCP App (HF Ready)
# ==========================================================

import os, json, asyncio, traceback, time, threading, requests, socket
import nest_asyncio, streamlit as st
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI
import uvicorn

# ==========================================================
# 🔧 Load Environment and Setup
# ==========================================================
load_dotenv()
nest_asyncio.apply()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ==========================================================
# 🌐 Utility: Find Free Port
# ==========================================================
def get_free_port(start_port=8000, end_port=9000):
    """Return the first available port in the given range."""
    for port in range(start_port, end_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available")

# ==========================================================
# ⚙️ Setup MCP (No Server Needed)
# ==========================================================
from mcp.server.fastmcp import FastMCP
import traceback

# Create MCP instance
mcp = FastMCP("UnifiedLawMCP")

# -----------------------------
# Setup LLM (ChatGroq)
# -----------------------------
try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
except Exception:
    llm = None

# -----------------------------
# Optional DuckDuckGo search
# -----------------------------
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except Exception:
    HAS_DDG = False

# -----------------------------
# Helper: Run LLM or Mock
# -----------------------------
def run_llm_or_mock(prompt: str) -> str:
    try:
        if llm:
            resp = llm.invoke(prompt)
            return getattr(resp, "content", str(resp))
    except Exception as e:
        return f"[LLM error] {e}\n\nPrompt:\n{prompt}"
    return f"[Mock response] Prompt:\n\n{prompt}"

# ==========================================================
# 📘 MCP Legal Tools (same as before)
# ==========================================================
@mcp.tool()
def rti_info(facts: str) -> str:
    prompt = (
        "You are an expert on Indian RTI (Right to Information) process.\n\n"
        f"FACTS: {facts}\n\n"
        "List relevant Acts/sections, the procedure to file an RTI (portals/forms/fees), "
        "and provide a short sample RTI application template with placeholders."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def divorce_info(facts: str) -> str:
    prompt = (
        "You are an expert on family law & divorce in India.\n\n"
        f"FACTS: {facts}\n\n"
        "List likely legal provisions/sections, procedural steps (mediation, court filings), "
        "documents required, timelines, and provide a short sample petition template."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def consumer_complaint_info(facts: str) -> str:
    prompt = (
        "You are an expert on consumer protection in India.\n\n"
        f"FACTS: {facts}\n\n"
        "List relevant portions of the Consumer Protection Act, how to approach consumer forums, "
        "documents to attach, approximate fees, and sample complaint structure."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def property_dispute_info(facts: str) -> str:
    prompt = (
        "You are an expert on property law in India.\n\n"
        f"FACTS: {facts}\n\n"
        "Identify relevant statutes/sections, likely remedies (civil suit, injunction, possession), "
        "documents to gather, and next-step recommendations."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def workplace_issue_info(facts: str) -> str:
    prompt = (
        "You are an expert on Indian labour & employment law.\n\n"
        f"FACTS: {facts}\n\n"
        "Identify applicable statutes/sections (payment, termination, POSH if applicable), "
        "procedural steps, and an action plan (letters, conciliation, complaints)."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def family_law_info(facts: str) -> str:
    prompt = (
        "You are an expert on family law in India (maintenance, custody, adoption, guardianship).\n\n"
        f"FACTS: {facts}\n\n"
        "List relevant legal provisions, likely steps, documents, and suggested next actions."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def cybercrime_info(facts: str) -> str:
    prompt = (
        "You are an expert on cybercrime law in India (IT Act, IPC supplements).\n\n"
        f"FACTS: {facts}\n\n"
        "List likely offences and sections, steps for preservation of evidence, how to file a police complaint/FIR, "
        "and online portals to report cybercrime."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def guide_steps(case_type: str) -> str:
    prompt = (
        f"You are a legal expert in Indian law. Provide a clear, step-by-step practical guide for a '{case_type}' case. "
        "Include relevant Acts, Sections, and Rules (e.g., RTI Act, 2005; Consumer Protection Act, 2019; Hindu Marriage Act, 1955) "
        "and explain how it's relevant to the case. Also mention required documents, online portals, fees, and timelines."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def draft_letter(case_type: str, facts: str) -> str:
    prompt = (
        f"Draft a formal, editable legal letter for '{case_type}' based on these facts:\n\n{facts}\n\n"
        "Include placeholders like [Name], [Date], [Recipient]. "
        "Ensure the draft cites at least one relevant law or section (e.g., Section 6(1) of the RTI Act, 2005)."
    )
    return run_llm_or_mock(prompt)

@mcp.tool()
def web_search(query: str) -> str:
    if HAS_DDG:
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=5)
            formatted = []
            for r in results:
                t = r.get("title") or "No title"
                href = r.get("href") or r.get("url") or ""
                body = r.get("body") or ""
                formatted.append(f"- {t}\n  {href}\n  {body}")
            return "\n\n".join(formatted) if formatted else "No results found."
        except Exception as e:
            return f"Search error: {e}"
    else:
        return f"[Search unavailable] DuckDuckGo not installed. Query was: {query}"

# ==========================================================
# ✅ Force MCP Tool Registration (Hugging Face Safe)
# ==========================================================
def ensure_mcp_tools_registered():
    """Guarantee that all MCP tools exist, even on Streamlit reload."""
    global mcp
    try:
        # If tools already exist, just log them
        if hasattr(mcp, "tools") and len(mcp.tools) > 0:
            print(f"✅ MCP has {len(mcp.tools)} tools: {list(mcp.tools.keys())}")
            return

        # If missing, manually rebuild the registry
        if not hasattr(mcp, "tools") or not isinstance(mcp.tools, dict):
            mcp.tools = {}

        # Rebind decorated functions (in case Streamlit reloaded)
        for name, obj in globals().items():
            if callable(obj) and hasattr(obj, "__wrapped__"):
                mcp.tools[name] = {"func": obj}
        print(f"🔁 Re-registered {len(mcp.tools)} MCP tools manually.")

    except Exception as e:
        print(f"⚠️ MCP registration recovery failed: {e}")
        mcp.tools = {}

# Run the registration immediately
ensure_mcp_tools_registered()

# ==========================================================
# ✅ Ensure MCP tools are registered before Streamlit loads
# ==========================================================
try:
    if hasattr(mcp, "tools"):
        print(f"✅ MCP initialized with tools")
    else:
        raise AttributeError("FastMCP.tools not created yet — forcing registration.")
except Exception as e:
    print(f"⚠️ MCP preload check failed: {e}")
    mcp.tools = getattr(mcp, "tools", {})

# ==========================================================
# 🚀 Local MCP Client (No server/port)
# ==========================================================

async def run_mcp_tool(tool_name: str, **payload):
    """Directly execute any MCP tool by name."""
    if not hasattr(mcp, "tools") or tool_name not in mcp.tools:
        return f"❌ Tool '{tool_name}' not found or MCP not initialized."

    try:
        tool_func = mcp.tools[tool_name]["func"]
        result = await asyncio.to_thread(tool_func, **payload)
        return result
    except Exception as e:
        return f"⚠️ MCP tool error: {e}\n{traceback.format_exc()}"



# ==========================================================
# 💬 Streamlit UI (Simplified — No FastAPI / No free_port)
# ==========================================================
st.set_page_config(page_title="⚖️ Nyaya Setu", layout="wide")
st.title("⚖️ Nyaya Setu")

with st.sidebar:
    st.header("Settings")
    MODEL = st.selectbox("Groq Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], index=0)
    TARGET_LANG = st.selectbox("Response Language", ["English", "Hindi", "Tamil", "Telugu"], index=0)

    if st.button("🧹 Clear Chat"):
        for key in ["history", "domains", "client_initialized", "tools"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.caption("🚀 Optimized for concurrency, context memory & faster responses")

if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not found.")
    st.stop()

# ==========================================================
# ⚖️ Core Logic
# ==========================================================
RAG_SOURCES = {
    "RTI": [
        "https://rti.gov.in",                 # Central RTI information portal
        "https://cic.gov.in",                 # Central Information Commission decisions
        "https://pgportal.gov.in"             # Online grievance redressal under RTI & others
    ],

    "Divorce": [
        "https://districts.ecourts.gov.in",   # Case filing & cause list
        "https://indiacode.nic.in",           # Statutory texts (HMA 1955, SMA 1954, etc.)
        "https://nalsa.gov.in",               # Legal aid & family dispute mediation
        "https://legalserviceindia.com"       # For case law summaries (secondary)
    ],

    "Consumer": [
        "https://consumerhelpline.gov.in",    # National consumer helpline
        "https://edaakhil.nic.in",            # Online complaint filing
        "https://ncdrc.nic.in"                # National Commission for redressal orders
    ],

    "Property": [
        "https://bhulekh.gov.in",             # Land records (generic portal)
        "https://igrmaharashtra.gov.in",      # State-specific property registration
        "https://dharani.telangana.gov.in",   # Modern state-level property records
        "https://revenue.gov.in"              # Central revenue department for property-related rules
    ],

    "Workplace": [
        "https://labour.gov.in",              # Ministry of Labour & Employment
        "https://epfindia.gov.in",            # Provident Fund & benefits
        "https://esic.gov.in",                # Employee State Insurance Corporation
        "https://poshforum.in"                # POSH Act awareness & local committee info
    ],

    "Family": [
        "https://ncw.nic.in",                 # National Commission for Women
        "https://ncpcr.gov.in",               # Child Rights Commission
        "https://wcd.nic.in",                 # Ministry of Women & Child Development
        "https://sci.gov.in"                  # Supreme Court (family law judgments)
    ],

    "Cybercrime": [
        "https://cybercrime.gov.in",          # National Cybercrime Reporting Portal
        "https://mha.gov.in",                 # Ministry of Home Affairs (BNS/BNSS updates)
        "https://cert-in.org.in",             # Indian Computer Emergency Response Team (incident handling)
        "https://cyberpeace.org"              # NGO partner recognized for awareness & training
    ],

    "Verdict Lookup": [
        "https://indiankanoon.org",           # Searchable case law database
        "https://legislative.gov.in",         # Gazette of India – acts & amendments
        "https://nalsa.gov.in",               # Legal services verdicts
        "https://main.sci.gov.in",            # Official Supreme Court judgments
        "https://highcourtchd.gov.in"         # Example state high court judgments (add others similarly)
    ],

    "Domestic Violence": [
        "https://ncw.nic.in",                 # Complaint & guidance under PWDVA
        "https://nalsa.gov.in",               # Legal aid for domestic violence survivors
        "https://wcd.nic.in",                 # Schemes & helplines
        "https://indiacode.nic.in"            # Acts: PWDVA 2005, IPC/BNSS sections
    ]
}

for key, default in {
    "history": [], "domains": None, "client_initialized": False, "tools": None
}.items():
    st.session_state.setdefault(key, default)

model = ChatGroq(model=MODEL, groq_api_key=GROQ_API_KEY)
client = None

async def init_tools():
    """Initialize MCP tools locally (no client/server)."""
    if not st.session_state.client_initialized:
        with st.spinner("🔗 Initializing local MCP tools..."):
            try:
                # Ensure FastMCP has the 'tools' attribute
                if not hasattr(mcp, "tools") or not isinstance(mcp.tools, dict):
                    raise AttributeError("FastMCP has no registered tools yet.")

                available = list(mcp.tools.keys())
                if not available:
                    raise ValueError("No MCP tools found. Check @mcp.tool() registrations.")

                st.session_state.tools = available
                st.session_state.client_initialized = True
                st.success(f"✅ Loaded {len(available)} MCP tools locally")
            except Exception as e:
                st.error(f"⚠️ MCP tool initialization failed: {e}")
                st.stop()



def translate_text(text, target="English"):
    if target == "English":
        return text
    try:
        return GoogleTranslator(source="auto", target=target.lower()).translate(text)
    except Exception:
        return text

def detect_domains(query: str):
    try:
        prompt = (
            "You are a classifier that identifies which of the following Indian legal domains "
            "the user's query belongs to.\n"
            f"Available domains: {list(RAG_SOURCES.keys())}\n\n"
            "Return ONLY a JSON list of matching domains. If none apply, return ['General'].\n\n"
            f"User query: {query}\n\nReturn JSON only."
        )
        resp = model.invoke(prompt)
        output = resp.content.strip()
        try:
            domains = json.loads(output)
            if isinstance(domains, list) and all(isinstance(d, str) for d in domains):
                return domains
        except Exception:
            pass
        query_lower = query.lower()
        keyword_map = {
             "marriage": "Divorce", "dowry": "Family",
            "salary": "Workplace", "harassment": "Workplace",
            "cyber": "Cybercrime", "online scam": "Cybercrime",
            "consumer": "Consumer", "refund": "Consumer",
            "property": "Property", "land": "Property",
            "tenant": "Property", "rti": "RTI", "information": "RTI",
            "court": "Verdict Lookup", "verdict": "Verdict Lookup"
        }
        matched = [v for k, v in keyword_map.items() if k in query_lower]
        return matched if matched else ["General"]
    except Exception:
        return ["General"]

async def analyze_case_with_tools(query):
    """Use MCP tools directly without HTTP client."""
    try:
        results = []
        for tool_name, tool_meta in mcp.tools.items():
            tool_func = tool_meta["func"]
            if "facts" in tool_func.__code__.co_varnames:
                result = await asyncio.to_thread(tool_func, facts=query)
                results.append(f"🧩 **{tool_name}**:\n{result}")
        return "\n\n".join(results[:3])
    except Exception as e:
        return f"⚠️ Tool analysis skipped: {e}"

async def generate_final_response(query, context_summary, domains):
    rag_info = f"Relevant domains: {', '.join(domains)} | Key references: {', '.join(sum([RAG_SOURCES.get(d, []) for d in domains], []))}"
    system_prompt = (
        "You are a helpful Indian legal assistant who provides concise, law-based answers grounded in Indian Acts and Sections.\n"
        "Always structure responses as:\n\n"
        "1. **Legal Basis (Laws & Sections)**\n"
        "2. **Recommended Steps**\n"
        "3. **Important Notes**\n"
        "4. **Draft Template (if applicable)**\n\n"
        f"{rag_info}\n"
    )

    messages = st.session_state.history[-4:] + [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": f"Context summary: {context_summary}"},
        {"role": "user", "content": query},
    ]

    response = await model.ainvoke(messages)
    return response.content.strip()

# ==========================================================
# 🧠 Chat Flow
# ==========================================================
user_input = st.chat_input("Describe your issue:")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    translated = GoogleTranslator(source="auto", target="en").translate(user_input)

    if st.session_state.domains is None:
        st.session_state.domains = detect_domains(translated)
        st.info(f"🧩 Detected domains: {', '.join(st.session_state.domains)}")

    domains = st.session_state.domains

    async def main_flow():
        await init_tools()
        tools = st.session_state.tools
        with st.spinner("🔍 Analyzing case with MCP tools..."):
            analysis = await analyze_case_with_tools(translated)
        with st.spinner("🧠 Drafting structured legal advice..."):
            final_resp = await generate_final_response(translated, analysis, domains)
        return final_resp

    final_output = asyncio.run(main_flow())
    # old
    final_output = asyncio.run(main_flow())

# new
    loop = asyncio.get_event_loop()
    if loop.is_running():
        final_output = loop.run_until_complete(main_flow())
    else:
        final_output = asyncio.run(main_flow())

    translated_final = translate_text(final_output, TARGET_LANG)

    st.session_state.history.append({"role": "assistant", "content": translated_final})
    st.chat_message("assistant").markdown(translated_final)

    st.download_button(
        "💾 Download Summary",
        data=translated_final,
        file_name="legal_summary.txt",
        mime="text/plain",
    )
