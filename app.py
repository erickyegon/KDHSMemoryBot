import os
import streamlit as st
import asyncio
import base64
from datetime import datetime
from pydantic import SecretStr
import pandas as pd
from PIL import Image
import io

# Import our new modules
from config import settings
from logger import log
from embedding import AsyncEuriaiEmbeddings
from chat import AsyncEuriaiChat
from vector_store import QdrantVectorStore
from conversation_store import ConversationStore
from memory_chatbot import MemoryChatbot
from pdf_processor import PDFProcessor
from pdf_retriever import PDFRetriever
from utils import format_date, test_embeddings, save_user_identity, mask_api_key, UserInput, MemoryInput, IdentityInput

# Page configuration
st.set_page_config(
    page_title=settings.app_name,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "initialized" not in st.session_state:
    log.info("Initializing session state")
    st.session_state.initialized = True
    st.session_state.chat_history = []
    st.session_state.suggestions = []
    st.session_state.pdf_mode = False
    st.session_state.loaded_pdfs = []
    st.session_state.current_pdf = None
    st.session_state.pdf_search_results = []

# API Key check
if not settings.euriai_api_key.get_secret_value():
    st.error("⚠️ Euriai API key not found. Please add it to your .env file or provide it below:")
    api_key = st.text_input("Euriai API Key", type="password")
    if api_key:
        os.environ["EURIAI_API_KEY"] = api_key
        settings.euriai_api_key = SecretStr(api_key)
        st.success("API key saved for this session!")

        # Test the API key
        with st.spinner("Testing API key..."):
            test_result = test_embeddings(api_key)

        if test_result["success"]:
            st.success("✅ API key is valid! Embeddings test successful.")
            st.json(test_result)
        else:
            st.error(f"❌ API key test failed: {test_result['error']}")
            st.stop()

        st.rerun()
    st.stop()

# Initialize chatbot if not already in session state
if "chatbot" not in st.session_state:
    log.info("Initializing chatbot")
    embedding_model = AsyncEuriaiEmbeddings()
    chat_model = AsyncEuriaiChat()
    conversation_store = ConversationStore()
    vector_store = QdrantVectorStore(embedding_model=embedding_model)

    st.session_state.chatbot = MemoryChatbot(
        embedding_model=embedding_model,
        chat_model=chat_model,
        conversation_store=conversation_store,
        vector_store=vector_store
    )

# Initialize PDF processor and retriever
if "pdf_processor" not in st.session_state:
    log.info("Initializing PDF processor")
    st.session_state.pdf_processor = PDFProcessor()

if "pdf_retriever" not in st.session_state:
    log.info("Initializing PDF retriever")
    st.session_state.pdf_retriever = PDFRetriever(
        pdf_processor=st.session_state.pdf_processor,
        embedding_model=embedding_model
    )

    # Load existing PDFs
    st.session_state.loaded_pdfs = st.session_state.pdf_processor.get_pdf_list()

def display_message(role, content, timestamp=None):
    """Display a chat message."""
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
            if timestamp:
                st.caption(format_date(timestamp))
    else:
        with st.chat_message("assistant"):
            st.write(content)
            if timestamp:
                st.caption(format_date(timestamp))

def handle_user_input(user_input):
    """Process user input and generate a response."""
    try:
        # Validate input
        validated = UserInput(content=user_input)
        user_input = validated.content

        if user_input.strip():
            # Display user message
            display_message("user", user_input)

            # Add to session chat history for display
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })

            # Check if we're in PDF mode
            if st.session_state.pdf_mode:
                # Process as PDF query
                with st.spinner("Searching KDHS data..."):
                    response_obj = process_pdf_query(user_input)

                # Display assistant response
                display_message("assistant", response_obj["response"])

                # Add to session chat history for display
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_obj["response"],
                    "timestamp": datetime.now().isoformat()
                })

                # Display sources if available
                if response_obj["sources"]:
                    with st.expander("Sources from KDHS Report", expanded=True):
                        for i, source in enumerate(response_obj["sources"]):
                            st.markdown(f"**Source {i+1}:** {source['filename']}, Page {source['page']}")
                            st.markdown(f"*{source['text']}*")
                            st.markdown("---")
            else:
                # Process with regular chatbot
                with st.spinner("Thinking..."):
                    response_obj = st.session_state.chatbot.process_input(user_input)

                # Display assistant response
                display_message("assistant", response_obj["response"], response_obj["metadata"]["timestamp"])

                # Add to session chat history for display
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_obj["response"],
                    "timestamp": response_obj["metadata"]["timestamp"]
                })

                # Update suggestions
                st.session_state.suggestions = st.session_state.chatbot.get_suggestions()

            log.info(f"Processed user input: {user_input[:50]}...")
    except Exception as e:
        log.error(f"Error handling user input: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def load_conversation_history():
    """Load and display conversation history."""
    history = st.session_state.chatbot.get_conversation_history()

    if history:
        st.session_state.chat_history = history

        # Display existing messages
        for message in history:
            display_message(message["role"], message["content"], message.get("timestamp"))

        log.info(f"Loaded conversation history with {len(history)} messages")

def save_identity():
    """Save user identity."""
    if st.session_state.identity_input and st.session_state.identity_input.strip():
        try:
            # Validate input
            validated = IdentityInput(content=st.session_state.identity_input)
            identity = validated.content

            # Save to environment and settings
            success = save_user_identity(identity)

            # Update chatbot
            st.session_state.chatbot.set_user_identity(identity)

            if success:
                st.sidebar.success("Identity saved successfully!")
            else:
                st.sidebar.error("Failed to save identity.")

            log.info(f"Updated user identity: {identity[:50]}...")
        except Exception as e:
            log.error(f"Error saving identity: {str(e)}")
            st.sidebar.error(f"Error: {str(e)}")

def add_custom_memory():
    """Add a custom memory entry."""
    if st.session_state.memory_input and st.session_state.memory_input.strip():
        try:
            # Validate input
            validated = MemoryInput(content=st.session_state.memory_input)
            memory_text = validated.content

            # Add to vector store via memory chain
            memory_id = st.session_state.chatbot.add_to_memory(
                memory_text,
                {"type": "custom", "source": "user_input"}
            )

            if memory_id >= 0:
                st.sidebar.success(f"Memory added successfully! (ID: {memory_id})")
                # Clear memory input
                st.session_state.memory_input = ""

                # Update suggestions
                st.session_state.suggestions = st.session_state.chatbot.get_suggestions()

                log.info(f"Added custom memory: {memory_text[:50]}... with ID: {memory_id}")
            else:
                st.sidebar.error("Failed to add memory.")
        except Exception as e:
            log.error(f"Error adding memory: {str(e)}")
            st.sidebar.error(f"Error: {str(e)}")

def upload_pdf():
    """Handle PDF upload."""
    uploaded_file = st.session_state.pdf_upload

    if uploaded_file is not None:
        try:
            # Get file content
            file_content = uploaded_file.read()
            filename = uploaded_file.name

            # Add to PDF processor
            pdf_metadata = st.session_state.pdf_processor.add_pdf(file_content, filename)

            # Update loaded PDFs list
            st.session_state.loaded_pdfs = st.session_state.pdf_processor.get_pdf_list()

            # Set as current PDF
            st.session_state.current_pdf = filename

            # Process the PDF
            with st.spinner(f"Processing PDF {filename}..."):
                st.session_state.pdf_processor.process_pdf(filename)

            # Index the PDF for search
            with st.spinner(f"Indexing PDF {filename} for search..."):
                asyncio.run(st.session_state.pdf_retriever.index_pdf(filename))

            st.sidebar.success(f"PDF {filename} uploaded, processed, and indexed successfully!")

            # Switch to PDF mode
            st.session_state.pdf_mode = True

            log.info(f"Uploaded and processed PDF: {filename}")
        except Exception as e:
            log.error(f"Error uploading PDF: {str(e)}")
            st.sidebar.error(f"Error uploading PDF: {str(e)}")

def select_pdf():
    """Handle PDF selection."""
    selected_pdf = st.session_state.pdf_select

    if selected_pdf:
        st.session_state.current_pdf = selected_pdf
        st.session_state.pdf_mode = True
        log.info(f"Selected PDF: {selected_pdf}")

async def search_pdf(query: str):
    """Search PDF content for relevant information."""
    if not query or not query.strip():
        return []

    try:
        # Determine filter criteria
        filter_criteria = None
        if st.session_state.current_pdf:
            filter_criteria = {"filename": st.session_state.current_pdf}

        # Search for relevant content
        results = await st.session_state.pdf_retriever.search(
            query=query,
            k=5,
            filter_criteria=filter_criteria
        )

        log.info(f"Found {len(results)} results for query: {query}")
        return results
    except Exception as e:
        log.error(f"Error searching PDF: {str(e)}")
        return []

def process_pdf_query(query: str):
    """Process a query about PDF content."""
    try:
        # Search for relevant content
        results = asyncio.run(search_pdf(query))

        if not results:
            return {
                "response": "I couldn't find any relevant information in the PDF documents. Please try a different query or upload a relevant PDF.",
                "sources": []
            }

        # Store search results
        st.session_state.pdf_search_results = results

        # Format results for the prompt
        formatted_results = ""
        sources = []

        for i, result in enumerate(results):
            # Extract basic info
            text = result.get("text", "")
            filename = result.get("filename", "Unknown")
            page = result.get("page", 0)
            content_type = result.get("content_type", "text")

            # Format based on content type
            if content_type == "table":
                formatted_results += f"\nTABLE {i+1} (from {filename}, page {page}):\n{text}\n"
            else:
                formatted_results += f"\nTEXT {i+1} (from {filename}, page {page}):\n{text}\n"

            # Add to sources
            sources.append({
                "filename": filename,
                "page": page,
                "content_type": content_type,
                "text": text[:100] + "..." if len(text) > 100 else text
            })

            # Add context if available
            context = result.get("context", {})

            if context.get("table_data"):
                table_data = context["table_data"]
                formatted_results += f"\nAdditional table data for TABLE {i+1}:\n"
                formatted_results += f"Columns: {', '.join(table_data.get('columns', []))}\n"

            if context.get("section"):
                section = context["section"]
                formatted_results += f"\nSection for TEXT {i+1}: {section.get('title', '')}\n"

        # Create prompt for the LLM
        prompt = f"""
You are an assistant that answers questions based on the Kenya Demographic Health Survey (KDHS) data.
Answer the following question using ONLY the information provided in the excerpts below.
If the information needed is not in the excerpts, say "I don't have enough information to answer this question based on the KDHS data provided."
Do not make up or infer information that is not explicitly stated in the excerpts.

QUESTION: {query}

EXCERPTS FROM KDHS REPORT:
{formatted_results}

Provide a comprehensive answer with specific statistics and data points from the KDHS when available.
If there are tables, interpret the data in them clearly.
"""

        # Generate response
        response = st.session_state.chatbot.chat_model.generate_completion(prompt)

        return {
            "response": response,
            "sources": sources
        }
    except Exception as e:
        log.error(f"Error processing PDF query: {str(e)}")
        return {
            "response": f"I encountered an error while processing your query: {str(e)}",
            "sources": []
        }

# Sidebar
with st.sidebar:
    st.title(f"🧠 {settings.app_name}")

    # Mode selection
    mode_options = ["Memory Chat", "KDHS Data Query"]
    selected_mode = st.radio("Select Mode", mode_options, index=1 if st.session_state.pdf_mode else 0)

    # Update mode based on selection
    st.session_state.pdf_mode = (selected_mode == "KDHS Data Query")

    st.markdown("---")

    # Instructions section based on mode
    if st.session_state.pdf_mode:
        with st.expander("📚 KDHS Data Query Instructions", expanded=True):
            st.markdown("""
            ### What This Mode Can Do

            - **Query KDHS Reports**: Ask questions about Kenya Demographic Health Survey data
            - **Extract Tables & Graphs**: Retrieve and interpret tables and graphs from reports
            - **Cite Sources**: Provides page numbers and sources for all information
            - **Analyze Demographics**: Get insights on health indicators across different demographics

            ### How To Use

            1. **Upload KDHS PDF**: Use the upload section below to add KDHS reports
            2. **Select Report**: Choose which report to query from the dropdown
            3. **Ask Questions**: Type specific questions about the KDHS data
            4. **View Sources**: See exactly where information comes from in the report

            ### Example Questions

            - "What is the infant mortality rate in Kenya according to KDHS 2022?"
            - "How does contraceptive use vary by education level?"
            - "What percentage of children are fully vaccinated?"
            - "Show me maternal health indicators by county"
            """)
    else:
        with st.expander("📚 Memory Chat Instructions", expanded=True):
            st.markdown("""
            ### What This Mode Can Do

            - **Remember Conversations**: Recalls previous interactions and references them
            - **Store Custom Memories**: Add specific information you want the bot to remember
            - **Personalized Responses**: Adapts to your identity information
            - **Context-Aware Replies**: Uses relevant memories to provide better answers
            - **Suggest Related Information**: Shows memories related to your conversation

            ### How To Use

            1. **Chat Normally**: Type messages in the chat input at the bottom
            2. **Add Your Identity**: Use the Identity section below to tell the bot about yourself
            3. **Add Custom Memories**: Use the Memory section to add information you want remembered
            4. **View Suggestions**: Related memories appear below the chat
            """)

    st.markdown("---")

    # PDF upload and selection (only in PDF mode)
    if st.session_state.pdf_mode:
        st.subheader("KDHS Report Management")

        # Upload new PDF
        st.file_uploader("Upload KDHS Report (PDF)", type=["pdf"], key="pdf_upload", on_change=upload_pdf)

        # Select existing PDF
        if st.session_state.loaded_pdfs:
            pdf_options = [pdf["filename"] for pdf in st.session_state.loaded_pdfs]
            current_index = pdf_options.index(st.session_state.current_pdf) if st.session_state.current_pdf in pdf_options else 0

            st.selectbox(
                "Select KDHS Report",
                pdf_options,
                index=current_index,
                key="pdf_select",
                on_change=select_pdf
            )

            # Show current PDF info
            if st.session_state.current_pdf:
                current_pdf = None
                for pdf in st.session_state.loaded_pdfs:
                    if pdf["filename"] == st.session_state.current_pdf:
                        current_pdf = pdf
                        break

                if current_pdf:
                    st.info(f"""
                    **Active Report:** {current_pdf.get('title', current_pdf['filename'])}
                    **Pages:** {current_pdf.get('num_pages', 'Unknown')}
                    **Processed:** {'Yes' if current_pdf.get('processed', False) else 'No'}
                    """)
        else:
            st.info("No KDHS reports loaded. Please upload a PDF file.")

        st.markdown("---")

    # API key display (masked)
    api_key_masked = mask_api_key(settings.euriai_api_key.get_secret_value())
    st.success(f"✅ Using API key: {api_key_masked}")

    # Model information
    st.info(f"📝 Embedding model: {settings.embedding_model}\n🤖 Chat model: {settings.chat_model}")

    st.markdown("---")

    # User identity section
    st.subheader("User Identity")
    identity = st.session_state.chatbot.user_identity or "Not set"
    st.text_area("Current Identity", value=identity, height=100, disabled=True)

    st.text_area("Update Identity", key="identity_input",
                placeholder="Enter information about yourself...", height=150)
    st.button("Save Identity", on_click=save_identity)

    st.markdown("---")

    # Custom memory section
    st.subheader("Add Custom Memory")
    st.text_area("Memory Content", key="memory_input",
                placeholder="Enter a memory to store...", height=150)
    st.button("Add Memory", on_click=add_custom_memory)

    st.markdown("---")

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.chatbot.clear_conversation()
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
if st.session_state.pdf_mode:
    st.title("Kenya Demographic Health Survey (KDHS) Data Assistant")
else:
    st.title(f"Chat with {settings.app_name}")

# Load and display conversation history
if not st.session_state.chat_history:
    # Show welcome message if no conversation history
    if len(st.session_state.chatbot.get_conversation_history()) == 0:
        if st.session_state.pdf_mode:
            st.markdown("""
            ## 📊 Welcome to the KDHS Data Assistant!

            This assistant helps you query and analyze data from the Kenya Demographic Health Survey reports.

            ### Try asking:
            - "What is the infant mortality rate in Kenya according to KDHS 2022?"
            - "How does contraceptive use vary by education level?"
            - "What percentage of children are fully vaccinated?"
            - "Show me maternal health indicators by county"

            Upload a KDHS report PDF using the sidebar to get started!
            """)

            # Show PDF upload prompt if no PDFs loaded
            if not st.session_state.loaded_pdfs:
                st.info("⬅️ Please upload a KDHS report PDF using the sidebar to begin.")
        else:
            st.markdown("""
            ## 👋 Welcome to Memory Chatbot!

            This chatbot remembers your conversations and uses them to provide more personalized and contextual responses.

            ### Try asking:
            - "What can you help me with?"
            - "Remember that I prefer vegetarian food"
            - "What are my food preferences?"
            - "My favorite color is blue"
            - "What's my favorite color?"
            - "I work as a data scientist"
            - "Tell me about my job"

            Check the sidebar for instructions on how to use all features!
            """)

    load_conversation_history()
else:
    # Display existing messages from session
    for message in st.session_state.chat_history:
        display_message(message["role"], message["content"], message.get("timestamp"))

# Chat input
user_input = st.chat_input("Type your message here...")
if user_input:
    handle_user_input(user_input)

# Memory and Suggestions panel
st.markdown("---")

# Different panels based on mode
if st.session_state.pdf_mode:
    # Create tabs for KDHS data
    report_tab, search_tab = st.tabs(["📊 KDHS Report Content", "🔍 Search Results"])

    with report_tab:
        st.subheader("KDHS Report Content")

        if st.session_state.current_pdf:
            # Get PDF metadata
            current_pdf = None
            for pdf in st.session_state.loaded_pdfs:
                if pdf["filename"] == st.session_state.current_pdf:
                    current_pdf = pdf
                    break

            if current_pdf and current_pdf.get("processed", False):
                # Show PDF structure
                try:
                    # Get sections from the PDF
                    pdf_content = st.session_state.pdf_processor.get_pdf_content(
                        st.session_state.current_pdf,
                        content_type="sections"
                    )

                    if pdf_content:
                        # Organize sections by page
                        sections_by_page = {}
                        for page_num, sections in pdf_content.items():
                            if sections:
                                sections_by_page[page_num] = sections

                        if sections_by_page:
                            st.write("### Report Structure")
                            for page_num, sections in sorted(sections_by_page.items()):
                                with st.expander(f"Page {page_num}", expanded=False):
                                    for section in sections:
                                        st.markdown(f"- {section.get('title', 'Untitled Section')}")

                        # Show tables found in the document
                        tables = st.session_state.pdf_processor.get_pdf_content(
                            st.session_state.current_pdf,
                            content_type="tables"
                        )

                        if tables:
                            st.write("### Tables in Report")
                            table_count = sum(len(page_tables) for page_tables in tables.values())
                            st.info(f"Found {table_count} tables in the report")

                            # Show sample tables
                            sample_tables = []
                            for page_num, page_tables in tables.items():
                                for table in page_tables[:2]:  # Limit to 2 tables per page
                                    sample_tables.append((page_num, table))
                                    if len(sample_tables) >= 5:  # Show at most 5 tables
                                        break
                                if len(sample_tables) >= 5:
                                    break

                            for page_num, table in sample_tables:
                                with st.expander(f"Table on Page {page_num}", expanded=False):
                                    # Try to convert to DataFrame for display
                                    try:
                                        if table.get("data") and table.get("columns"):
                                            df = pd.DataFrame(table["data"])
                                            st.dataframe(df)
                                    except Exception as e:
                                        st.write(f"Could not display table: {str(e)}")
                                        st.write(table.get("data", "No data available"))
                    else:
                        st.info("No structured content found in the report. Try asking questions directly.")
                except Exception as e:
                    st.error(f"Error displaying PDF content: {str(e)}")
            else:
                st.info("PDF has not been processed yet or processing information is not available.")
        else:
            st.info("No KDHS report selected. Please select or upload a report from the sidebar.")

    with search_tab:
        st.subheader("Search Results")

        if st.session_state.pdf_search_results:
            st.write(f"### Found {len(st.session_state.pdf_search_results)} relevant passages")

            for i, result in enumerate(st.session_state.pdf_search_results):
                # Extract basic info
                text = result.get("text", "")
                filename = result.get("filename", "Unknown")
                page = result.get("page", 0)
                content_type = result.get("content_type", "text")
                similarity = result.get("similarity", 0)

                with st.expander(f"Result {i+1} from {filename}, Page {page} (Relevance: {similarity:.2f})", expanded=i==0):
                    st.markdown(f"**Content Type:** {content_type}")
                    st.markdown(f"**Text:** {text}")

                    # Show images if available
                    context = result.get("context", {})
                    if context.get("images"):
                        st.write("**Related Images:**")
                        for img in context["images"][:2]:  # Limit to 2 images
                            try:
                                image_path = img.get("path")
                                if image_path and os.path.exists(image_path):
                                    image = Image.open(image_path)
                                    st.image(image, caption=f"Image from page {page}", width=400)
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
        else:
            st.info("No search results yet. Ask a question about the KDHS data to see results here.")
else:
    # Create tabs for memory chatbot
    memory_tab, suggestions_tab = st.tabs(["📚 Stored Memories", "💡 Suggestions"])

    with memory_tab:
        st.subheader("What I Remember About You")

        # Get all memories from the vector store
        all_memories = st.session_state.chatbot.vector_store.get_all_memories()

        if all_memories:
            # Sort memories by timestamp (newest first)
            all_memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            # Display memories in a table
            for i, memory in enumerate(all_memories[:10]):  # Show only the 10 most recent memories
                memory_text = memory.get("text", "")
                memory_type = memory.get("type", "unknown")
                memory_time = format_date(memory.get("timestamp", ""))

                with st.expander(f"Memory {i+1}: {memory_text[:50]}...", expanded=False):
                    st.markdown(f"**Full Text:** {memory_text}")
                    st.markdown(f"**Type:** {memory_type}")
                    st.markdown(f"**Added:** {memory_time}")

            if len(all_memories) > 10:
                st.info(f"Showing 10 of {len(all_memories)} total memories. The chatbot uses all memories when responding.")
        else:
            st.info("No memories stored yet. Start chatting or add custom memories to build up the memory database.")

    with suggestions_tab:
        st.subheader("Suggestions & Related Memories")

        if st.session_state.suggestions:
            for i, suggestion in enumerate(st.session_state.suggestions):
                with st.expander(f"Suggestion {i+1} (Relevance: {suggestion['relevance']:.2f})"):
                    st.markdown(suggestion["text"])
        else:
            st.info("No suggestions available yet. Continue the conversation to generate relevant suggestions.")
