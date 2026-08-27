from pathlib import Path

from research_agent.retrieval.chunking import WordWindowChunker
from research_agent.retrieval.models import ResearchDocument


def make_document(text: str, name: str = "paper.md") -> ResearchDocument:
    print(ResearchDocument.from_text(source_path=Path(name), doc_type="md", text=text))
    return ResearchDocument.from_text(source_path=Path(name), doc_type="md", text=text)


def test_chunker_creates_stable_overlapping_windows() -> None:
    document = make_document("zero one two three four five six seven eight nine")
    chunker = WordWindowChunker(window_size=4, overlap=1)
    print("chunker\n",chunker)

    chunks = chunker.split([document])
    print
    for chunk in chunks:
        print("chunk tesxt\n", chunk.text)

    # assert [chunk.text for chunk in chunks] == [
    #     "zero one two three",
    #     "three four five six",
    #     "six seven eight nine",
    # ]
    # assert [chunk.doc_id for chunk in chunks] == [document.doc_id] * 3
    # assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

make_document("hello my name is Adya")
test_chunker_creates_stable_overlapping_windows()
