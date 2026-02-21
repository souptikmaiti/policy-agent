# Policy Agent

A RAG-powered AI agent built with [BeeAI Framework](https://github.com/i-am-bee/bee-agent-framework) and [AgentStack SDK](https://github.com/AgentStack-AI/AgentStack) that helps users understand healthcare insurance policies by answering questions based on uploaded policy documents.

## Overview

The Policy Agent uses Retrieval-Augmented Generation (RAG) to:
- Extract text from uploaded insurance policy PDFs
- Index policy documents in a vector database
- Answer questions based on retrieved policy information
- Provide accurate coverage details and policy explanations
- Recommend suitable policies based on treatment costs

## Features

- **📄 PDF Processing**: Automatic text extraction from policy documents
- **🔍 Semantic Search**: Vector-based search for relevant policy information
- **💬 Multi-turn Conversations**: Maintains context across multiple questions
- **🤖 AI-Powered Responses**: Uses Google Gemini for intelligent answers
- **📊 Real-time Streaming**: Streams responses and trajectory updates to the UI
- **💾 Persistent Storage**: Vector store persists across conversation sessions

## Architecture

```
User Upload PDF → Extract Text → Chunk Document → Generate Embeddings → Store Vectors
                                                                              ↓
User Query → Embed Query → Search Vector Store → Retrieve Context → LLM Response
```

### Key Components

- **BeeAI RequirementAgent**: Core agent with tool execution
- **UnconstrainedMemory**: Session-based conversation memory
- **AgentStack VectorStore**: Managed vector database for document chunks
- **OpenAI Embeddings**: Via AgentStack embedding extension
- **LangChain Text Splitter**: Markdown-aware document chunking
- **Trajectory Extension**: Real-time UI updates

## Prerequisites

- Python 3.13+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- UV package manager ([Install UV](https://docs.astral.sh/uv/))

## Installation

1. **Clone the repository**
```bash
cd A2A/healthcare-assistant/policy-agent
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

3. **Install dependencies**
```bash
uv sync
```

## Usage

### Local Development

```bash
uv run policy_agent
```

The agent will start on `http://localhost:8004`

### Docker Deployment

```bash
docker build -t policy-agent .
docker run -p 8004:8004 --env-file .env policy-agent
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | Yes |
| `PRODUCTION_MODE` | Enable production mode | No (default: false) |

### Agent Configuration

Edit `policy_agent/agentstack_policy.py` to customize:

**LLM Settings**:
```python
llm_client = GeminiChatModel(
    model_id="gemini-2.0-flash-exp",
    parameters=ChatModelParameters(temperature=0.3),  # Adjust creativity
    client_options={"api_key": google_api_key}
)
```

**Chunking Strategy**:
```python
def chunk_markdown(markdown_text: str) -> list[str]:
    splitter = MarkdownTextSplitter(
        chunk_size=1000,      # Adjust chunk size
        chunk_overlap=200     # Adjust overlap
    )
    return splitter.split_text(markdown_text)
```

**Search Parameters**:
```python
results = await search_vector_store(
    vector_store, query, embedding_client, embedding_model,
    limit=5  # Number of chunks to retrieve
)
```

## How It Works

### 1. Document Processing Pipeline

When a user uploads a PDF:

```python
# Extract text from PDF
extraction = await file.create_extraction()
async with file.load_text_content() as loaded_file:
    markdown_text = loaded_file.text

# Split into chunks
chunks = chunk_markdown(markdown_text)

# Store in vector database
items = await vector_store.add_documents(items=chunks)
```

### 2. Query Processing

When a user asks a question:

```python
# Search vector store
results = await search_vector_store(
    vector_store, query, embedding_client, embedding_model
)

# Create search tool with retrieved context
def policy_search(search_query: str) -> str:
    return "\n\n".join([f"[Relevance: {res.score:.2f}]\n{res.text}" for res in results])

# Run agent with tool
async for event, meta in agent.run(query):
    if meta.name == "final_answer":
        yield event.delta
```

### 3. Conversation Persistence

Vector store ID is saved in conversation context:

```python
# Save vector store ID
await context.store(AgentMessage(parts=[Part(data={"vector_store_id": vector_store.id})]))

# Retrieve in next message
async for message in context.load_history():
    if "vector_store_id" in message.parts[0].data:
        vector_store = await VectorStore.get(message.parts[0].data["vector_store_id"])
```

## Example Interactions

**Upload Policy**:
```
User: [Uploads health_insurance_policy.pdf]
Agent: ✅ Indexing Complete - Stored 47 document chunks
```

**Ask Questions**:
```
User: "What is covered under outpatient care?"
Agent: "Based on the policy document, outpatient care coverage includes:
- Doctor consultations (80% coverage)
- Diagnostic tests (70% coverage)
- Prescription medications (60% coverage)
Annual limit: $5,000"
```

**Follow-up Questions**:
```
User: "What about emergency room visits?"
Agent: "Emergency room visits are covered at 90% with no prior authorization required.
Co-pay: $100 per visit. Annual limit: $25,000"
```

## Development

### Project Structure

```
policy-agent/
├── policy_agent/
│   └── agentstack_policy.py  # Main agent implementation
├── pyproject.toml            # Dependencies
├── Dockerfile                # Container configuration
├── .env.example              # Environment template
└── README.md                 # This file
```

### Dependencies

- `beeai-framework[a2a]==0.1.74`: Core agent framework
- `agentstack-sdk==0.4.3`: AgentStack integration
- `agent-framework-a2a==1.0.0b251120`: A2A protocol support
- `openai>=1.0.0`: Embedding client
- `langchain-text-splitters>=0.3.0`: Document chunking

## Troubleshooting

### Embedding Extension Missing
**Error**: "Embedding extension not provided"
**Solution**: Ensure the AgentStack platform provides embedding fulfillment

### Extraction Failed
**Error**: "Extraction failed with status: failed"
**Solution**: 
- Verify PDF is not password-protected
- Check file size limits
- Ensure PDF contains extractable text (not scanned images)

### Vector Store Not Found
**Problem**: Agent doesn't remember uploaded documents
**Solution**: Check `context.load_history()` is working and vector store ID is being saved

### No Results Found
**Problem**: Search returns empty results
**Solution**:
- Verify documents were successfully indexed
- Try broader search queries
- Check embedding model compatibility

## API Response Format

The agent streams responses in real-time:

```python
# Trajectory updates
yield trajectory.trajectory_metadata(
    title="Searching Policy Documents...",
    content=f"Query: {user_query}"
)

# Final answer
yield AgentMessage(text="Based on the policy...")
```

## Performance Optimization

**Chunk Size Tuning**:
- Smaller chunks (500-800): Better precision, more API calls
- Larger chunks (1200-1500): Better context, fewer API calls

**Search Limit**:
- Lower limit (3-5): Faster, more focused
- Higher limit (7-10): More comprehensive, slower

**Temperature**:
- Lower (0.1-0.3): More factual, consistent
- Higher (0.5-0.7): More creative, varied

## Contributing

Contributions welcome! Key areas:
- Support for more document formats (DOCX, TXT)
- Advanced chunking strategies
- Multi-document search
- Citation tracking
- Performance optimizations

## License

[Add your license here]

## Contact

**Maintainer**: Your Name  
**Email**: your-email@example.com

## Related Agents

- [ConciergeAgent](../concierge-agent/) - Main healthcare assistant
- [HealthcareResearchAgent](../research-agent/) - Disease and treatment research
- [HospitalsAgent](../hospitals-agent/) - Hospital search
- [DoctorFinderAgent](../doctors-agent/) - Doctor search

## Acknowledgments

- Built with [BeeAI Framework](https://github.com/i-am-bee/bee-agent-framework)
- Powered by [AgentStack SDK](https://github.com/AgentStack-AI/AgentStack)
- Uses [Google Gemini](https://ai.google.dev/) for LLM
- Document processing via [LangChain](https://www.langchain.com/)