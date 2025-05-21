# Memory Chatbot: Production-Grade Conversational AI with Memory Capabilities

## Executive Summary

Memory Chatbot is an enterprise-ready, production-grade conversational AI system with dual capabilities:

1. **Memory-Based Chat**: Leverages advanced vector search technology to maintain context and provide personalized responses based on conversation history.

2. **KDHS Data Assistant**: A specialized multimodal RAG (Retrieval-Augmented Generation) system that can process and answer questions about Kenya Demographic Health Survey (KDHS) reports, including text, tables, and graphs.

The application implements sophisticated retrieval mechanisms using vector embeddings to create a system that can both "remember" past interactions and extract relevant information from complex PDF documents. This demonstrates expertise in modern AI architecture, multimodal data processing, and enterprise-grade system design.

### Key Technical Achievements

- **Vector-Based Memory System**: Implemented using FAISS and Qdrant with automatic fallback mechanisms
- **Multimodal PDF Processing**: Extracts and indexes text, tables, and images from complex reports
- **Asynchronous Architecture**: Non-blocking API calls for high throughput and responsiveness
- **Production-Ready Infrastructure**: Containerization, monitoring, logging, and security features
- **Resilient Design**: Graceful degradation, error recovery, and comprehensive exception handling
- **Enterprise Integration**: Configurable for integration with existing authentication systems and APIs

This project showcases expertise in building production-grade AI applications that are scalable, maintainable, and ready for enterprise deployment, with particular strength in multimodal data processing and retrieval-augmented generation.

## Technical Overview

### Core Technologies

- **Python 3.10+**: Modern Python features including type hints, async/await, and context managers
- **Vector Databases**: Qdrant (primary) with FAISS (fallback) for similarity search
- **Embedding Models**: Integration with state-of-the-art embedding models via Euriai API
- **PDF Processing**: Advanced extraction of text, tables, and images from complex documents
- **OCR & Table Extraction**: Pytesseract, Tabula, and Camelot for comprehensive document parsing
- **Streamlit**: Interactive web interface with real-time updates
- **Docker & Docker Compose**: Containerization for consistent deployment
- **Pydantic**: Type-safe configuration and data validation
- **Loguru**: Structured, configurable logging system
- **HTTPX**: Asynchronous HTTP client for non-blocking API calls

### Architecture

The application follows a modular, service-oriented architecture with clear separation of concerns:

#### Memory Chat Mode
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Web Interface  │────▶│  Memory Chatbot │────▶│  Vector Store   │
│   (Streamlit)   │     │    (Core Logic) │     │  (Qdrant/FAISS) │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │   API Clients   │
                        │ (Async/Fallback)│
                        │                 │
                        └─────────────────┘
```

#### KDHS Data Assistant Mode
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Web Interface  │────▶│  PDF Processor  │────▶│  PDF Retriever  │
│   (Streamlit)   │     │  (Extraction)   │     │  (RAG System)   │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │                 │     │                 │
                        │ Table & Image   │     │  Vector Store   │
                        │   Processing    │     │  (Qdrant/FAISS) │
                        │                 │     │                 │
                        └─────────────────┘     └─────────────────┘
```

#### Key Components

- **Configuration System** (`config.py`): Centralized, type-safe configuration using Pydantic
- **Logging System** (`logger.py`): Structured logging with rotation and multiple outputs
- **Embedding Client** (`embedding.py`): Asynchronous client for generating text embeddings
- **Chat Client** (`chat.py`): Asynchronous client for generating chat completions
- **Vector Store** (`vector_store.py`): Manages vector database with automatic fallback
- **Conversation Store** (`conversation_store.py`): Persists and manages conversation history
- **Memory Chatbot** (`memory_chatbot.py`): Core business logic for memory-based chat
- **PDF Processor** (`pdf_processor.py`): Extracts text, tables, and images from PDF documents
- **PDF Retriever** (`pdf_retriever.py`): Indexes and retrieves relevant content from PDFs
- **Utilities** (`utils.py`): Input validation, security functions, and helper methods
- **Web Interface** (`app.py`): Streamlit-based user interface with dual-mode functionality

### Advanced Features

#### 1. Intelligent Memory Retrieval

The system uses a sophisticated algorithm to retrieve relevant memories:

```python
async def process_input_async(self, user_input: str) -> Dict[str, Any]:
    # Validate input
    validated_input = UserInput(content=user_input)
    user_input = validated_input.content

    # Add user message to conversation history
    self.conversation_store.add_user_message(user_input)

    # Get conversation history for context
    formatted_history = self.conversation_store.get_formatted_history()

    # Get relevant memories using vector similarity search
    relevant_memories = self.vector_store.search(user_input, settings.memory_k)

    # Format memories and create prompt with context
    formatted_memories = self._format_memories(relevant_memories)
    prompt = self._create_prompt(user_input, formatted_history, formatted_memories)

    # Generate response with context-aware prompt
    response = await self.chat_model.generate_completion_async(prompt)

    # Store interaction in memory for future reference
    await self._add_interaction_to_memory_async(user_input, response)

    return response_obj
```

#### 2. Fault Tolerance and Graceful Degradation

The system implements multiple fallback mechanisms:

- **Vector Database Fallback**: Automatically switches to FAISS if Qdrant is unavailable
- **API Client Fallback**: Tries multiple endpoints and methods if primary fails
- **Asynchronous Retry Logic**: Implements exponential backoff for transient failures
- **Comprehensive Error Handling**: Provides meaningful error messages and recovery paths

#### 3. Multimodal PDF Processing

The system implements sophisticated PDF processing capabilities:

```python
def process_pdf(self, filename: str, max_pages: int = None) -> Dict[str, Any]:
    """Process a PDF file to extract text, tables, and images."""
    # Extract text and structure using unstructured
    elements = partition_pdf(filepath, extract_images_in_pdf=True)

    # Process elements by type
    for element in elements:
        element_type = type(element).__name__

        if hasattr(element, "text") and element.text.strip():
            # Store text with metadata
            processed_data["text"][page_num].append({
                "text": element.text,
                "type": element_type,
                "page": page_num,
                "metadata": element.metadata.__dict__
            })

    # Extract tables using tabula and camelot
    tables = tabula.read_pdf(filepath, pages=pages_to_process, multiple_tables=True)

    # Extract images
    images = convert_from_path(filepath, first_page=min(pages_to_process),
                              last_page=max(pages_to_process))
```

Key capabilities include:
- **Text Extraction**: Extracts structured text with metadata
- **Table Detection**: Identifies and parses tables into structured data
- **Image Extraction**: Extracts charts, graphs, and images for analysis
- **Document Structure**: Maintains document hierarchy and section information
- **OCR Processing**: Applies OCR to extract text from images when needed

#### 4. Security Features

- **Input Validation**: All user inputs are validated and sanitized using Pydantic models
- **Secrets Management**: API keys and sensitive data are handled securely
- **Data Sanitization**: Prevents injection attacks and malicious inputs
- **Containerized Isolation**: Docker containers provide security boundaries

## Implementation Details

### Vector Database Implementation

The system uses a dual-database approach for maximum reliability:

#### Primary: Qdrant Vector Database

Qdrant is a high-performance vector similarity search engine that provides:

- **Scalability**: Handles millions of vectors efficiently
- **Filtering**: Supports complex metadata filtering alongside vector search
- **Persistence**: Reliable storage with transaction support
- **Clustering**: Supports distributed deployment for high availability

Implementation highlights:

```python
def _search_qdrant(self, query_embedding, k):
    """Search for memories in Qdrant."""
    try:
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )

        results = []
        for scored_point in search_result:
            memory = scored_point.payload.copy()
            memory["similarity"] = float(scored_point.score)
            results.append(memory)

        log.info(f"Found {len(results)} relevant memories in Qdrant")
        return results

    except Exception as e:
        log.error(f"Error searching Qdrant: {str(e)}")
        # Fallback to FAISS if Qdrant fails
        return self._search_faiss(query_embedding, k)
```

#### Fallback: FAISS Local Index

FAISS (Facebook AI Similarity Search) provides a robust local fallback:

- **Zero Dependencies**: Works without external services
- **Efficiency**: Optimized C++ implementation with Python bindings
- **Algorithms**: Implements multiple indexing strategies for different needs
- **Local Operation**: Functions without network connectivity

The fallback mechanism ensures the system remains operational even when the primary database is unavailable:

```python
def _initialize_faiss_fallback(self):
    """Initialize a local FAISS index as fallback."""
    import faiss
    self.use_fallback = True
    self.index_path = f"data/faiss_index"
    self.metadata_path = f"{self.index_path}_metadata.json"

    # Load existing index or create new one
    if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            log.info(f"Loaded existing FAISS index with {len(self.metadata)} entries")
        except Exception as e:
            log.error(f"Error loading FAISS index: {str(e)}. Creating new index.")
            self._create_new_faiss_index()
    else:
        self._create_new_faiss_index()
```

### Asynchronous API Integration

The system implements non-blocking API calls for improved performance:

```python
async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of documents asynchronously."""
    if not texts:
        return []

    # Process in batches for large text collections
    batch_size = 20
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks = []
        for batch in batches:
            payload = {
                "input": batch,
                "model": self.model
            }

            task = client.post(
                self.embed_url,
                headers=self._get_headers(),
                json=payload
            )
            tasks.append(task)

        # Wait for all requests to complete concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses and handle errors
        all_embeddings = []
        for response in responses:
            if isinstance(response, Exception):
                log.error(f"Error in async embedding request: {str(response)}")
                continue

            try:
                response.raise_for_status()
                data = response.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                log.error(f"Error processing embedding response: {str(e)}")

    return all_embeddings
```

### Configuration Management

The application uses Pydantic for type-safe configuration:

```python
class Settings(BaseSettings):
    """Application settings using Pydantic for validation and type safety."""

    # API Keys
    api_key: SecretStr = Field(
        default=SecretStr(""),
        env="EURIAI_API_KEY",
        description="API key for Euriai services"
    )

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL",
        description="Model to use for embeddings"
    )

    # Vector Store Settings
    vector_dim: int = Field(
        default=1536,
        env="VECTOR_DIM",
        description="Dimension of embedding vectors"
    )

    # Database Settings
    qdrant_host: str = Field(
        default="localhost",
        env="QDRANT_HOST",
        description="Qdrant server host"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
```

### Structured Logging

The application implements comprehensive logging with loguru:

```python
# Add console handler
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Add file handler with rotation
logger.add(
    settings.log_file,
    rotation="10 MB",
    retention="1 month",
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)
```

## Performance Considerations

### Memory Optimization

The system implements several memory optimization strategies:

1. **Batch Processing**: Processes embeddings in batches to avoid memory spikes
2. **Stream Processing**: Uses streaming responses where available
3. **Garbage Collection**: Explicit garbage collection for large operations
4. **Memory Monitoring**: Logs memory usage for performance tracking

### Scalability

The application is designed for horizontal scalability:

1. **Stateless Design**: Core components are stateless for easy replication
2. **Database Separation**: Vector database can be scaled independently
3. **Container Orchestration**: Ready for Kubernetes deployment
4. **Load Balancing**: Supports multiple instances behind a load balancer

### Caching Strategy

The system implements strategic caching:

1. **Embedding Cache**: Frequently used embeddings are cached to reduce API calls
2. **Response Cache**: Common responses are cached with appropriate TTL
3. **Metadata Cache**: Vector search metadata is cached separately from vectors
4. **Cache Invalidation**: Implements proper invalidation strategies

## Installation and Deployment

### Prerequisites

#### Core Requirements
- **Python 3.10+**: Required for modern language features
- **Docker & Docker Compose**: For containerized deployment (optional)
- **Euriai API Key**: For embedding and chat completion APIs
- **50MB Disk Space**: Minimum for application and dependencies
- **2GB RAM**: Recommended for optimal performance

#### PDF Processing Requirements (for KDHS Data Assistant)
- **Tesseract OCR**: Required for image-to-text conversion
- **Java Runtime Environment (JRE)**: Required for Tabula-py
- **Poppler**: Required for PDF2Image
- **200MB Additional Disk Space**: For PDF processing libraries
- **4GB RAM**: Recommended for processing large PDFs

### Local Development Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/memory-chatbot.git
   cd memory-chatbot
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   # Option 1: Core dependencies only
   pip install -r requirements.txt

   # Option 2: Core + PDF processing dependencies
   pip install -r requirements-full.txt

   # Option 3: Using the setup script (recommended)
   python setup.py
   ```

   For PDF processing capabilities, install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr poppler-utils default-jre

   # macOS
   brew install tesseract poppler openjdk

   # Windows
   # Download and install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
   # Download and install Poppler: https://github.com/oschwartz10612/poppler-windows/releases
   # Download and install JRE: https://www.oracle.com/java/technologies/downloads/
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```
   EURIAI_API_KEY="your-api-key-here"
   USER_IDENTITY="Information about the user or organization"
   EMBEDDING_MODEL="text-embedding-3-small"
   CHAT_MODEL="gpt-4.1-nano"
   LOG_LEVEL="INFO"
   ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

1. **Build and Start Containers**:
   ```bash
   # Build and start in detached mode
   docker-compose up -d

   # View logs
   docker-compose logs -f
   ```

2. **Access the Application**:
   Open your browser and navigate to `http://localhost:8501`

3. **Stop the Application**:
   ```bash
   docker-compose down
   ```

### Production Deployment

For production environments, consider the following deployment options:

#### Option 1: Docker with Nginx Reverse Proxy

1. **Create Docker Network**:
   ```bash
   docker network create memory-chatbot-network
   ```

2. **Deploy Nginx Reverse Proxy**:
   ```bash
   docker run -d \
     --name nginx-proxy \
     --network memory-chatbot-network \
     -p 80:80 -p 443:443 \
     -v /path/to/certs:/etc/nginx/certs \
     -v /path/to/nginx.conf:/etc/nginx/conf.d/default.conf \
     nginx:latest
   ```

3. **Deploy Memory Chatbot**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

#### Option 2: Kubernetes Deployment

1. **Apply Kubernetes Manifests**:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secret.yaml
   kubectl apply -f k8s/qdrant-deployment.yaml
   kubectl apply -f k8s/qdrant-service.yaml
   kubectl apply -f k8s/memory-chatbot-deployment.yaml
   kubectl apply -f k8s/memory-chatbot-service.yaml
   kubectl apply -f k8s/ingress.yaml
   ```

2. **Verify Deployment**:
   ```bash
   kubectl get pods -n memory-chatbot
   kubectl get services -n memory-chatbot
   ```

#### Option 3: Cloud Platform Deployment

The application can be deployed to major cloud platforms:

- **AWS**: Using ECS, ECR, and RDS
- **Azure**: Using AKS, ACR, and Azure Database
- **GCP**: Using GKE, GCR, and Cloud SQL

Detailed deployment guides for each platform are available in the `docs/deployment` directory.

## Usage Guide

The application offers two distinct modes: Memory Chat and KDHS Data Assistant.

### Memory Chat Mode

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Select Memory Chat Mode**:
   - Choose "Memory Chat" from the mode selector in the sidebar

3. **Configure User Identity**:
   - Enter your identity information in the sidebar
   - Click "Save Identity"

4. **Chat with the Bot**:
   - Type messages in the chat input
   - View responses with timestamps
   - See relevant memories and suggestions below

5. **Adding Custom Memories**:
   - Navigate to the "Add Custom Memory" section in the sidebar
   - Type or paste the memory text
   - Click "Add Memory"
   - Verify with the success message showing memory ID

### KDHS Data Assistant Mode

1. **Select KDHS Data Assistant Mode**:
   - Choose "KDHS Data Query" from the mode selector in the sidebar

2. **Upload KDHS Report**:
   - Use the file uploader in the sidebar to upload a KDHS PDF report
   - Wait for processing to complete (this may take a few minutes for large reports)

3. **Browse Report Content**:
   - View the report structure in the "KDHS Report Content" tab
   - Explore sections, tables, and images extracted from the document

4. **Query the Data**:
   - Type specific questions about the KDHS data in the chat input
   - For example: "What is the infant mortality rate in Kenya according to KDHS 2022?"
   - View the response with citations to specific pages and sections

5. **View Sources**:
   - Examine the sources panel that appears below responses
   - See exactly which parts of the document were used to generate the answer
   - View relevant tables and visualizations from the report

### Advanced Usage

#### API Integration

The Memory Chatbot can be integrated with other applications via API:

```python
import requests

# Initialize a session
session = requests.Session()

# Add a memory
def add_memory(text):
    response = session.post(
        "http://localhost:8000/api/memories",
        json={"text": text}
    )
    return response.json()

# Chat with the bot
def chat(message):
    response = session.post(
        "http://localhost:8000/api/chat",
        json={"message": message}
    )
    return response.json()

# Example usage
memory_id = add_memory("The user prefers vegetarian food.")
response = chat("What kind of food do I like?")
print(response["response"])
```

#### Batch Processing

For processing large amounts of data:

```bash
# Import memories from a CSV file
python scripts/import_memories.py --file data/memories.csv --column text

# Export conversation history
python scripts/export_conversations.py --output history.json --format json
```

## Advanced Configuration

### Custom Embedding Models

The system supports custom embedding models:

```python
# In .env file
EMBEDDING_MODEL="custom-model-name"
EMBEDDING_DIMENSION=768  # Adjust to match your model's output dimension

# Or in code
from memory_chatbot import MemoryChatbot
from embedding import AsyncEuriaiEmbeddings

custom_embeddings = AsyncEuriaiEmbeddings(
    model="custom-model-name",
    vector_dim=768
)

chatbot = MemoryChatbot(embedding_model=custom_embeddings)
```

### Database Tuning

Fine-tune the vector database for specific use cases:

```python
# In .env file
QDRANT_DISTANCE_METRIC="Cosine"  # Options: Cosine, Euclid, Dot
QDRANT_OPTIMIZE_FOR="Recall"     # Options: Recall, Performance

# Or in code
from vector_store import QdrantVectorStore

vector_store = QdrantVectorStore(
    distance_metric="Cosine",
    optimize_for="Recall",
    ef_construct=512,  # Higher values improve recall at the cost of indexing speed
    m=16               # Number of connections per element
)
```

### Memory Management

Configure memory retention and retrieval:

```python
# In .env file
MEMORY_K=10                # Number of memories to retrieve
MEMORY_THRESHOLD=0.7       # Minimum similarity threshold
MEMORY_RETENTION_DAYS=90   # How long to keep memories

# Or in code
chatbot = MemoryChatbot(
    memory_k=10,
    memory_threshold=0.7,
    memory_retention_days=90
)
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Verify API key in `.env` file
   - Check network connectivity
   - Ensure API endpoints are accessible

2. **Vector Database Issues**:
   - Verify Qdrant is running (`docker ps`)
   - Check Qdrant logs (`docker logs qdrant`)
   - Ensure proper port mapping

3. **Memory Usage Problems**:
   - Reduce batch size for large datasets
   - Increase container memory limits
   - Monitor memory usage with logging

### Logging and Debugging

Enable detailed logging for troubleshooting:

```
# In .env file
LOG_LEVEL="DEBUG"
LOG_FILE="logs/debug.log"
```

View logs:
```bash
# View application logs
tail -f logs/app.log

# View Docker logs
docker-compose logs -f
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Qdrant Team**: For their excellent vector database
- **Streamlit Team**: For the interactive web framework
- **Euriai**: For providing the embedding and chat completion APIs
- **Open Source Community**: For the various libraries that made this project possible
