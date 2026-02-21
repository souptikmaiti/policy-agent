from langchain_text_splitters import MarkdownTextSplitter


def chunk_markdown(markdown_text: str) -> list[str]:
    return MarkdownTextSplitter().split_text(markdown_text)
