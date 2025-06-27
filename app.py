import streamlit as st
import os
from rag_model import cover_letter_model
from pymilvus import MilvusClient
from dotenv import load_dotenv
import json

load_dotenv()

# --- Streamlit UI setup ---
st.set_page_config(page_title="UK Sponsor Licence Cover Letter Generator", layout="centered")
st.title("📄 UK Sponsor Licence Cover Letter Generator")

# --- Milvus/Zilliz client setup ---
@st.cache_resource

def get_milvus_client():
    return MilvusClient(uri=os.getenv("ZILLIZ_ADDRESS"), token=os.getenv("ZILLIZ_TOKEN"))

client = get_milvus_client()

# --- User input ---
st.markdown("""
Enter details about your company and why you're applying for a UK Skilled Worker sponsor licence. 
The assistant will generate a formal, UKVI-compliant cover letter using stored examples.
""")

user_input = st.text_area("✍️ Enter your request:", height=250, placeholder="E.g., We are a software firm in Manchester with 30 staff. We need to sponsor a data scientist under SOC code 2136 due to...", key="user_input")

if st.button("🚀 Generate Cover Letter") and user_input:
    with st.spinner("Generating letter using RAG model..."):
        try:
            response = cover_letter_model(client, user_input)
            if response:
                st.subheader("📬 Drafted Cover Letter")

                st.code(response, language="markdown")

                # 🔐 Escape safely for JavaScript using json.dumps
                escaped_response = json.dumps(response)

                # copy_code = f"""
                # <button onclick="navigator.clipboard.writeText({escaped_response})" style="
                #     background-color: #4CAF50;
                #     border: none;
                #     color: white;
                #     padding: 6px 12px;
                #     text-align: center;
                #     text-decoration: none;
                #     font-size: 14px;
                #     margin-top: 5px;
                #     border-radius: 5px;
                #     cursor: pointer;
                # ">📋 Copy to Clipboard</button>
                # """
                # st.markdown(copy_code, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.caption("Built with 💼 by UK LegalTech • Powered by LangChain, OpenAI, and Zilliz Milvus")
