import os
import json
import logging
from dotenv import load_dotenv
from typing import Annotated, Any
from pydantic import BaseModel, Field

from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.types import StringToolOutput, ToolRunOptions
from beeai_framework.context import RunContext as BeeRunContext
from beeai_framework.emitter import Emitter

from agentstack_sdk.platform import File, VectorStore, PlatformFileUrl
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from agentstack_sdk.a2a.extensions.ui.agent_detail import EnvVar, AgentDetailContributor
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.a2a.extensions import (
    AgentDetail, AgentDetailTool,
    TrajectoryExtensionServer, TrajectoryExtensionSpec,
    LLMServiceExtensionServer, LLMServiceExtensionSpec,
    EmbeddingServiceExtensionServer, EmbeddingServiceExtensionSpec,
    PlatformApiExtensionServer, PlatformApiExtensionSpec
)
from agentstack_sdk.server import Server
from a2a.types import DataPart, FilePart, FileWithUri, Message, Part, TextPart, AgentSkill

from .embedding.client import get_embedding_client
from .embedding.embed import embed_chunks
from .extraction import extract_file
from .text_splitting import chunk_markdown
from .vector_store.create import create_vector_store
from .vector_store.search import search_vector_store

load_dotenv()

logger = logging.getLogger("policy-agent")
logging.getLogger().setLevel(logging.DEBUG)


# File formats supported by the text-extraction service (docling)
default_input_modes = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
    "text/markdown",  # Markdown
    "text/asciidoc",  # AsciiDoc
    "text/html",  # HTML
    "application/xhtml+xml",  # XHTML
    "text/csv",  # CSV
    "image/png",  # PNG
    "image/jpeg",  # JPEG
    "image/tiff",  # TIFF
    "image/bmp",  # BMP
    "image/webp",  # WEBP
]

POLICY_INSTRUCTIONS = """You are a healthcare insurance policy assistant. Your role is to:

1. Answer questions about insurance policies, coverage, and claims based on retrieved documents
2. Provide accurate information from the policy documents
3. Help users understand their coverage options
4. Recommend suitable policies based on treatment costs
5. Explain policy terms and conditions clearly

Always use the policy_search tool to retrieve relevant information before answering."""


server = Server()

class PolicySearchToolInput(BaseModel):
    query: str = Field(description="Search query to find relevant information in policy documents")

class PolicySearchTool(Tool[PolicySearchToolInput, ToolRunOptions, StringToolOutput]):
    name = "policy_search" # type: ignore
    description = "Search insurance policy documents for coverage, terms, and conditions" # type: ignore
    input_schema = PolicySearchToolInput # type: ignore

    def __init__(self, vector_store: VectorStore, embedding_client, embedding_model, options: dict[str, Any] | None = None):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        super().__init__(options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "search", "retrieval", "vector_store"],
            creator=self,
        )
    
    async def _run(self, input: PolicySearchToolInput, options: ToolRunOptions | None, context: BeeRunContext) -> StringToolOutput:
        results = await search_vector_store(
            self.vector_store, input.query, self.embedding_client, self.embedding_model
        )
        # for res in results:
        #     print(f"Relevance: {res.score:.2f}\n{res.item.text}")
        snippet = "\n\n".join([f"[Relevance: {res.score:.2f}]\n{res.item.text}" for res in results])
        logger.info(f"Vector Search Results: {snippet[:500]}")
        return StringToolOutput(snippet)
    


@server.agent(
    name="PolicyAgent",
    default_input_modes=default_input_modes,
    default_output_modes=["text", "text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="Welcome! I can help you understand insurance policies. Upload a policy PDF or ask questions.",
        input_placeholder="Ask about insurance policies or upload a PDF...",
        programming_language="Python",
        framework="BeeAI",
        contributors=[
            AgentDetailContributor(name="Souptik Maiti", email="https://github.com/souptikmaiti"),
        ],
        variables=[
            EnvVar(name="GOOGLE_API_KEY", description="Google API Key", required=True),
            EnvVar(name="VECTOR_STORE_ID", description="Vector Store ID", required=False)
        ],
        tools=[
            AgentDetailTool(name="policy_search", description="Search policy documents for relevant information")
        ]
    ),
    skills=[
        AgentSkill(
            id="PolicyAgent",
            name="Healthcare Insurance Policy Agent", 
            description="Search policy documents for relevant information",
            tags=["Healthcare", "Insurance", "Policy", "Agent", "RAG"],
            examples=[
                "What is covered under outpatient care?",
                "What about emergency room visits?",
                "How much does it cost to see a specialist?"
            ]
        ),
    ]
)
async def policy_agent_wrapper(
    input: Message,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("gemini:gemini-3-flash-preview",)),
    ],
    embedding: Annotated[EmbeddingServiceExtensionServer, EmbeddingServiceExtensionSpec.single_demand()],
    platform: Annotated[PlatformApiExtensionServer, PlatformApiExtensionSpec()],
):
    """Policy Agent with RAG"""

    yield trajectory.trajectory_metadata(
        title="Initializing Agent...",
        content="Setting up Policy Agent with RAG"
    )

    # Get embedding client
    embedding_client, embedding_model = get_embedding_client(embedding)
    
    # Extract files and query from input
    files: list[File] = []
    query = ""
    for part in input.parts:
        match part.root:
            case FilePart(file=file_obj) if file_obj:
                # Extract file ID from URI
                file_id = PlatformFileUrl(file_obj.uri).file_id
                files.append(await File.get(file_id))
            case TextPart(text=text):
                query = text
            case _:
                raise NotImplementedError(f"Unsupported part: {type(part.root)}")

    vector_store = None
    VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", None)
    # Get global vector store by fixed ID
    if VECTOR_STORE_ID:
        vector_store = await VectorStore.get(VECTOR_STORE_ID)
        if vector_store:
            yield trajectory.trajectory_metadata(
                title="Vector Store Loaded",
                content=f"Using existing shared database with ID: {vector_store.id}"
            )

    # Check if vector store exists in context
    # async for message in context.load_history():
    #     match message:
    #         case Message(parts=[Part(root=DataPart(data=data))]):
    #             vector_store = await VectorStore.get(data["vector_store_id"])

    # Create vector store if it doesn't exist
    if not vector_store:
        yield trajectory.trajectory_metadata(
            title="Creating Vector Store...",
            content="Initializing vector store for policy documents"
        )
        vector_store = await create_vector_store(embedding_client, embedding_model)
        # store vector store id in context for future messages
        data_part = DataPart(data={"vector_store_id": vector_store.id})
        await context.store(AgentMessage(parts=[data_part]))
        yield trajectory.trajectory_metadata(
            title="Vector Store Created",
            content=f"Created vector store with ID: {vector_store.id} \n Run: agentstack env add PolicyAgent VECTOR_STORE_ID={vector_store.id}"
        )
        # set manually via
        # agentstack env add PolicyAgent VECTOR_STORE_ID="vector-store-id from response above"

    # Process files and add to vector store
    for file in files:
        yield trajectory.trajectory_metadata(
            title="Processing File...",
            content=f"Extracting text from {file.filename}"
        )
        
        await extract_file(file)
        
        async with file.load_text_content() as loaded_file:
            chunks = chunk_markdown(loaded_file.text)
            
            yield trajectory.trajectory_metadata(
                title="Chunking Document...",
                content=f"Created {len(chunks)} chunks"
            )
            
            items = await embed_chunks(file, chunks, embedding_client, embedding_model)
            await vector_store.add_documents(items=items)

            yield trajectory.trajectory_metadata(
                title="Indexing Complete",
                content=f"Stored {len(items)} document chunks"
            )

    # If query exists, search vector store
    if query:
        yield trajectory.trajectory_metadata(
            title="Searching Policy Documents...",
            content=f"Query: {query}"
        )
        
        search_tool = PolicySearchTool(vector_store, embedding_client, embedding_model)
        
        # Configure LLM from extension fulfillment
        if not llm or not llm.data:
            yield trajectory.trajectory_metadata(title="LLM Error", content="LLM extension missing.")
            yield "LLM selection is required."
            return

        llm_config = llm.data.llm_fulfillments.get("default")
        if not llm_config:
            yield trajectory.trajectory_metadata(title="LLM Error", content="No LLM fulfillment available.")
            yield "No LLM configuration available from the extension."
            return
        
        # logger.info(f"LLM Config: {llm_config}")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            yield trajectory.trajectory_metadata(title="LLM Error", content="GOOGLE_API_KEY environment variable is required.")
            yield "GOOGLE_API_KEY environment variable is required."
            return
        
        llm_client = GeminiChatModel(
            model_id=llm_config.api_model.removeprefix("gemini:").removeprefix("models/"),
            parameters=ChatModelParameters(temperature=0.3, max_tokens=4096, stream=True),
            api_key=google_api_key,
        )
        
        memory = UnconstrainedMemory()
        
        agent = RequirementAgent(
            llm=llm_client,
            memory=memory,
            tools=[search_tool],
            instructions=POLICY_INSTRUCTIONS,
            role="Insurance Policy Assistant",
            requirements=[
                ConditionalRequirement(search_tool, min_invocations=1, max_invocations=1)
            ],
        )
        
        yield trajectory.trajectory_metadata(
            title="Generating Response...",
            content="Analyzing policy information"
        )

        response_text = ""

        def handle_final_answer_stream(data, meta) -> None:
            nonlocal response_text
            # Accumulate streamed final answer text
            if getattr(data, "delta", None):
                response_text += data.delta
        
        def summarize_for_trajectory(data: object, limit: int = 400) -> str:
            """
            Convert tool inputs/outputs to a readable, bounded string for trajectory updates.
            """
            try:
                text = data if isinstance(data, str) else json.dumps(data, default=str)
            except Exception:
                text = str(data)

            return text if len(text) <= limit else f"{text[:limit]}... [truncated]"
        
        # Run agent and stream response
        async for event, meta in agent.run(
            query,
            execution=AgentExecutionConfig(max_iterations=20, max_retries_per_step=2)
        ).on("final_answer", handle_final_answer_stream):
            if meta.name == "final_answer":
                if getattr(event, "delta", None):
                    yield event.delta
                elif getattr(event, "text", None):
                    response_text += event.text
            elif meta.name == "success" and event.state.steps:
                step = event.state.steps[-1]
                tool_name = step.tool.name if step.tool else "Unknown Tool"
                trajectory.trajectory_metadata(
                    title="Tool Called",
                    content=f"Tool called: {tool_name}"
                )
                if step.tool and step.tool.name != "final_answer":
                    yield trajectory.trajectory_metadata(title=f"{tool_name} (request)", content=summarize_for_trajectory(step.input))

                    if getattr(step, "error", None):
                        yield trajectory.trajectory_metadata(title=f"{tool_name} (error)", content=step.error.explain())
                    else:
                        output_text = step.output.get_text_content() if getattr(step, "output", None) else "No output"
                        yield trajectory.trajectory_metadata(title=f"{tool_name} (response)", content=summarize_for_trajectory(output_text))

        # Emit final answer
        yield AgentMessage(text=response_text)

    elif files:
        yield AgentMessage(text=f"{len(files)} file(s) processed")
    else:
        yield AgentMessage(text="Nothing to do")


def run():
    server.run(
        host="0.0.0.0",
        port=8000,
        context_store=PlatformContextStore(),
    )


if __name__ == "__main__":
    run()
