from agentstack_sdk.platform import VectorStore, VectorStoreSearchResult
from google.genai import client
from google.genai import types


async def search_vector_store(
    vector_store: VectorStore,
    query: str,
    embedding_client: client.AsyncClient,
    embedding_model: str,
) -> list[VectorStoreSearchResult]:
    embedding_response = await embedding_client.models.embed_content(
        model=embedding_model,
        contents=query,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    
    if not embedding_response.embeddings or not embedding_response.embeddings[0].values:
        raise ValueError("Failed to generate query embedding")
    
    query_vector = embedding_response.embeddings[0].values
    return await vector_store.search(query_vector=query_vector, limit=5)
