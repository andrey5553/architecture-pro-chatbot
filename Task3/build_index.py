#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import time
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class KnowledgeBaseIndexer:

    DB_DIR_POSTFIX = "faiss_db\\"
    STATS_FILENAME = "indexing_stats.json"

    def __init__(self, setup_dir: str | None = None):
        self.current_dir = setup_dir or os.path.dirname(os.path.abspath(__file__))
        self.faiss_db_dir = os.path.join(self.current_dir, self.DB_DIR_POSTFIX) #os.path.join("e:\\", self.DB_DIR_POSTFIX)
        self.stats_file = os.path.join(self.current_dir, self.STATS_FILENAME)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},  # Используем CPU для совместимости
            encode_kwargs={"normalize_embeddings": True},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200
        )
        self.processing_time = float(0)

        self.documents = list[Document]()
        self.chunks = list[Document]()

    def load_documents(
        self, knowledge_base_path: str, filter_short_chunks=True
    ) -> list[Document]:
        for address, _, files in os.walk(knowledge_base_path):
            for filename in files:
                file_path = os.path.join(address, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if not content:
                        continue

                    relative_path = file_path.replace(knowledge_base_path, "")

                    metadata = {
                        "source": str(relative_path),
                        "filename": filename,
                        "file_path": str(file_path),
                        "file_size": len(content),
                    }
                    doc = Document(page_content=content, metadata=metadata)
                    self.documents.append(doc)

                except Exception as error:
                    print(f"Failed reading {file_path}:", error)
        for doc in self.documents:
            chunks = self.text_splitter.split_documents([doc])
            filtered_chunks = []
            for i, chunk in enumerate(chunks):
                if filter_short_chunks and len(chunk.page_content.strip()) < 100:
                    continue

                chunk.metadata.update(
                    {
                        "chunk_id": f"{doc.metadata['filename']}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )
                filtered_chunks.append(chunk)

            self.chunks.extend(filtered_chunks)

        print(f"Loaded {len(self.documents)} docs with {len(self.chunks)} chunk")
        return self.chunks

    def create_faiss_index(self) -> FAISS:
        start_time = time.time()

        # Load existing storage is exists.
        # Add current documents anyway.
        if self.chunks and not os.path.isdir(self.faiss_db_dir):
            faiss_index = FAISS.from_documents(
                documents=[self.chunks[0]], embedding=self.embeddings
            )
            self.add_chunks_to_index(faiss_index, self.chunks[1:])

            os.makedirs(self.faiss_db_dir, exist_ok=True)
            faiss_index.save_local(self.faiss_db_dir)
        else:
            faiss_index = FAISS.load_local(
                folder_path=self.faiss_db_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

        self.processing_time += time.time() - start_time
        return faiss_index

    def add_chunks_to_index(self, faiss_index: FAISS, chunks: list[Document]):
        total = len(chunks)
        for i in tqdm(range(total), total=total, desc="Index FAISS"):
            faiss_index.add_documents([self.chunks[i]])

    def save_statistics(self, index: FAISS):
        stats = {
            "model": {
                "name": self.embeddings.model_name,
                "repository": f"https://huggingface.co/sentence-transformers/{self.embeddings.model_name}",
                "embeddings": json.loads(self.embeddings.model_dump_json()),
            },
            "knowledge_base": {
                "total_documents": len(self.documents),
                "total_chunks": len(self.chunks),
            },
            "faiss": {
                "persist_directory": self.DB_DIR_POSTFIX,
            },
            "processing": {
                "time_seconds": self.processing_time,
                "chunks_per_second": len(self.chunks) / self.processing_time,
            },
        }

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        return stats


if __name__ == "__main__":
    indexer = KnowledgeBaseIndexer()

    knowledge_base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../task2/knowledge_data"
    )
    indexer.load_documents(knowledge_base)

    index = indexer.create_faiss_index()
    stats = indexer.save_statistics(index)
    print("Indexing statistics:", stats)
