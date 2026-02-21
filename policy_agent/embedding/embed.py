from agentstack_sdk.platform import File, VectorStoreItem
from google.genai import client
from google.genai import types


async def embed_chunks(
    file: File, chunks: list[str], embedding_client: client.AsyncClient, embedding_model: str
) -> list[VectorStoreItem]:
    vector_store_items = []

    embedding_result = await embedding_client.models.embed_content(
        contents=chunks,
        model=embedding_model,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    if not embedding_result.embeddings:
        raise ValueError("No embeddings returned")
    for i, embedding_data in enumerate(embedding_result.embeddings):
        if not embedding_data.values:
            raise ValueError(f"No embedding values returned for chunk {i}")
        
        item = VectorStoreItem(
            document_id=file.id,
            document_type="platform_file",
            model_id=embedding_model,
            text=chunks[i],
            embedding=embedding_data.values,
            metadata={"chunk_index": str(i)},
        )
        vector_store_items.append(item)
    return vector_store_items
