from agentstack_sdk.a2a.extensions import EmbeddingServiceExtensionServer
from google.genai import client
from dotenv import load_dotenv
import os

load_dotenv()


def get_embedding_client(
    embedding: EmbeddingServiceExtensionServer,
) -> tuple[client.AsyncClient, str]:
    if not embedding or not embedding.data:
        raise ValueError("Embedding extension not provided")

    embedding_config = embedding.data.embedding_fulfillments.get("default")
    if not embedding_config:
        raise ValueError("Default embedding configuration not found")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    embedding_client = client.Client(api_key=google_api_key).aio
    embedding_model = embedding_config.api_model.removeprefix("gemini:").removeprefix("models/")
    return embedding_client, embedding_model
