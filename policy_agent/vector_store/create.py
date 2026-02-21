import os
from dotenv import load_dotenv
from agentstack_sdk.platform import VectorStore
from google.genai import client
from google.genai import types

load_dotenv()


async def create_vector_store(embedding_client: client.AsyncClient, embedding_model: str):
    result = await embedding_client.models.embed_content(
        model=embedding_model,
        contents="test",
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    
    if not result.embeddings or not result.embeddings[0].values:
        raise ValueError("Failed to get embedding dimension")
    
    dimension = len(result.embeddings[0].values)

    return await VectorStore.create(
        name="policy-documents",
        dimension=dimension,
        model_id=embedding_model,
        context_id='abcd',
    )
