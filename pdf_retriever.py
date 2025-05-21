import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from config import settings
from logger import log
from embedding import AsyncEuriaiEmbeddings
from vector_store import QdrantVectorStore
from pdf_processor import PDFProcessor

class PDFRetriever:
    """Retriever for PDF content using vector search."""
    
    def __init__(
        self,
        pdf_processor: Optional[PDFProcessor] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_model: Optional[AsyncEuriaiEmbeddings] = None,
        collection_name: str = "pdf_content"
    ):
        """Initialize the PDF retriever."""
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.embedding_model = embedding_model or AsyncEuriaiEmbeddings()
        
        # Create a dedicated vector store for PDF content
        self.vector_store = vector_store or QdrantVectorStore(
            embedding_model=self.embedding_model,
            collection_name=collection_name
        )
        
        # Track indexed documents
        self.indexed_docs = self._load_indexed_docs()
        
        log.info(f"Initialized PDFRetriever with {len(self.indexed_docs)} indexed documents")
    
    def _load_indexed_docs(self) -> Dict[str, Dict[str, Any]]:
        """Load information about indexed documents."""
        index_path = "data/pdf_index_metadata.json"
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                log.error(f"Error loading indexed docs metadata: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_indexed_docs(self):
        """Save information about indexed documents."""
        index_path = "data/pdf_index_metadata.json"
        
        try:
            with open(index_path, 'w') as f:
                json.dump(self.indexed_docs, f)
            log.info(f"Saved indexed docs metadata with {len(self.indexed_docs)} entries")
        except Exception as e:
            log.error(f"Error saving indexed docs metadata: {str(e)}")
    
    async def index_pdf(self, filename: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Index a PDF file for retrieval."""
        # Check if PDF exists
        pdf_list = self.pdf_processor.get_pdf_list()
        pdf_found = False
        pdf_metadata = None
        
        for pdf in pdf_list:
            if pdf["filename"] == filename:
                pdf_found = True
                pdf_metadata = pdf
                break
        
        if not pdf_found:
            raise ValueError(f"PDF {filename} not found")
        
        # Check if already indexed and not forcing reindex
        if filename in self.indexed_docs and not force_reindex:
            log.info(f"PDF {filename} already indexed. Use force_reindex=True to reindex.")
            return self.indexed_docs[filename]
        
        # Process PDF if not already processed
        if not pdf_metadata.get("processed", False):
            pdf_metadata = self.pdf_processor.process_pdf(filename)
        
        # Get processed data
        processed_data = pdf_metadata.get("processed_data", {})
        
        if not processed_data:
            raise ValueError(f"No processed data found for PDF {filename}")
        
        # Index text content
        text_chunks = []
        text_metadata = []
        
        # Process text by page
        for page_num, page_texts in processed_data.get("text", {}).items():
            for text_item in page_texts:
                text = text_item.get("text", "")
                
                # Skip empty text
                if not text or not text.strip():
                    continue
                
                # Create chunks (simplified - a more sophisticated chunking strategy would be better)
                chunks = self._chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    text_chunks.append(chunk)
                    
                    # Create metadata for the chunk
                    chunk_metadata = {
                        "filename": filename,
                        "title": pdf_metadata.get("title", filename),
                        "page": page_num,
                        "chunk_index": i,
                        "content_type": "text",
                        "text_type": text_item.get("type", "Text"),
                        "source": "pdf"
                    }
                    
                    text_metadata.append(chunk_metadata)
        
        # Index table content
        for page_num, page_tables in processed_data.get("tables", {}).items():
            for table_item in page_tables:
                # Convert table to text representation
                table_text = self._table_to_text(table_item)
                
                if table_text and table_text.strip():
                    text_chunks.append(table_text)
                    
                    # Create metadata for the table
                    table_metadata = {
                        "filename": filename,
                        "title": pdf_metadata.get("title", filename),
                        "page": page_num,
                        "content_type": "table",
                        "table_source": table_item.get("source", "unknown"),
                        "columns": table_item.get("columns", []),
                        "source": "pdf"
                    }
                    
                    text_metadata.append(table_metadata)
        
        # Generate embeddings for all chunks
        log.info(f"Generating embeddings for {len(text_chunks)} chunks from PDF {filename}")
        
        # Use async embedding for better performance
        embeddings = await self.embedding_model.embed_documents_async(text_chunks)
        
        # Add to vector store
        for i, (text, embedding, metadata) in enumerate(zip(text_chunks, embeddings, text_metadata)):
            self.vector_store.add_memory(
                text=text,
                metadata=metadata,
                embedding=embedding  # Pass pre-computed embedding
            )
        
        # Update indexed docs metadata
        self.indexed_docs[filename] = {
            "filename": filename,
            "title": pdf_metadata.get("title", filename),
            "num_pages": pdf_metadata.get("num_pages", 0),
            "num_chunks": len(text_chunks),
            "has_tables": pdf_metadata.get("has_tables", False),
            "has_images": pdf_metadata.get("has_images", False),
            "indexed_at": self._get_timestamp()
        }
        
        # Save indexed docs metadata
        self._save_indexed_docs()
        
        log.info(f"Indexed PDF {filename} with {len(text_chunks)} chunks")
        return self.indexed_docs[filename]
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find a good breaking point
            end = min(start + chunk_size, len(text))
            
            # Try to break at a paragraph or sentence
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind(". ", start, end),
                        text.rfind("! ", start, end),
                        text.rfind("? ", start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move start position with overlap
            start = end - overlap if end < len(text) else len(text)
        
        return chunks
    
    def _table_to_text(self, table_item: Dict[str, Any]) -> str:
        """Convert a table to a text representation."""
        try:
            columns = table_item.get("columns", [])
            data = table_item.get("data", [])
            
            if not columns or not data:
                return ""
            
            # Create a text representation of the table
            text = "Table:\n"
            
            # Add header
            text += " | ".join(str(col) for col in columns) + "\n"
            text += "-" * (sum(len(str(col)) for col in columns) + 3 * (len(columns) - 1)) + "\n"
            
            # Add rows
            for row in data:
                text += " | ".join(str(row.get(col, "")) for col in columns) + "\n"
            
            return text
        except Exception as e:
            log.error(f"Error converting table to text: {str(e)}")
            return ""
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def search(self, query: str, k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for relevant PDF content."""
        # Generate query embedding
        query_embedding = await self.embedding_model.embed_query_async(query)
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            k=k,
            query_embedding=query_embedding,
            filter_criteria=filter_criteria
        )
        
        # Enhance results with additional context
        enhanced_results = []
        
        for result in results:
            # Get the source PDF and page
            filename = result.get("filename")
            page = result.get("page")
            content_type = result.get("content_type", "text")
            
            # Get additional context if available
            context = {}
            
            if filename and page is not None:
                try:
                    # Get page content
                    page_content = self.pdf_processor.get_pdf_content(
                        filename=filename,
                        content_type="all",
                        page=page
                    )
                    
                    # Add relevant context based on content type
                    if content_type == "text":
                        # For text, add surrounding text
                        context["surrounding_text"] = self._get_surrounding_text(
                            page_content.get("text", []),
                            result.get("text", "")
                        )
                    
                    elif content_type == "table":
                        # For tables, add the original table data
                        context["table_data"] = self._find_matching_table(
                            page_content.get("tables", []),
                            result.get("columns", [])
                        )
                    
                    # Add any images on the page
                    if page_content.get("images"):
                        context["images"] = [
                            {
                                "path": img.get("path"),
                                "type": img.get("type")
                            }
                            for img in page_content.get("images", [])
                            if img.get("type") == "region"  # Only include region images, not full pages
                        ]
                    
                    # Add section information
                    if page_content.get("sections"):
                        context["section"] = self._find_relevant_section(
                            page_content.get("sections", [])
                        )
                except Exception as e:
                    log.error(f"Error getting additional context: {str(e)}")
            
            # Add context to result
            result["context"] = context
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _get_surrounding_text(self, page_texts: List[Dict[str, Any]], target_text: str) -> str:
        """Get text surrounding the target text."""
        # Find the text item containing the target text
        for text_item in page_texts:
            text = text_item.get("text", "")
            if target_text in text:
                # Return the full text item
                return text
        
        return ""
    
    def _find_matching_table(self, page_tables: List[Dict[str, Any]], target_columns: List[str]) -> Dict[str, Any]:
        """Find the table matching the target columns."""
        if not target_columns:
            return {}
        
        for table in page_tables:
            columns = table.get("columns", [])
            if set(target_columns).issubset(set(columns)):
                return table
        
        return {}
    
    def _find_relevant_section(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the most relevant section."""
        if not sections:
            return {}
        
        # For now, just return the first section
        # A more sophisticated approach would determine relevance
        return sections[0]
    
    def get_indexed_pdfs(self) -> List[Dict[str, Any]]:
        """Get a list of all indexed PDFs."""
        return list(self.indexed_docs.values())
    
    def get_image_base64(self, image_path: str) -> str:
        """Get base64 encoded image for display in UI."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                return encoded_string
        except Exception as e:
            log.error(f"Error encoding image: {str(e)}")
            return ""
