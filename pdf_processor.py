import os
import io
import re
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import pypdf
import pdfplumber
import tabula
import camelot
import cv2
from PIL import Image
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
import matplotlib.pyplot as plt
import plotly.express as px
from unstructured.partition.pdf import partition_pdf

from config import settings
from logger import log

# Create directories for storing processed data
os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/tables", exist_ok=True)
os.makedirs("data/text", exist_ok=True)

class PDFProcessor:
    """Process PDF files to extract text, tables, and images."""
    
    def __init__(self, pdf_dir: str = "data/pdfs"):
        """Initialize the PDF processor."""
        self.pdf_dir = pdf_dir
        self.loaded_pdfs = {}  # Store loaded PDF metadata
        
        # Load existing PDFs if available
        self._load_existing_pdfs()
        
        log.info(f"Initialized PDFProcessor with {len(self.loaded_pdfs)} existing PDFs")
    
    def _load_existing_pdfs(self):
        """Load metadata for existing PDFs in the directory."""
        if not os.path.exists(self.pdf_dir):
            os.makedirs(self.pdf_dir, exist_ok=True)
            return
            
        for filename in os.listdir(self.pdf_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.pdf_dir, filename)
                try:
                    # Get basic metadata without full processing
                    metadata = self._get_pdf_metadata(filepath)
                    self.loaded_pdfs[filename] = metadata
                except Exception as e:
                    log.error(f"Error loading PDF metadata for {filename}: {str(e)}")
    
    def _get_pdf_metadata(self, filepath: str) -> Dict[str, Any]:
        """Get basic metadata for a PDF file."""
        try:
            with open(filepath, "rb") as f:
                pdf = pypdf.PdfReader(f)
                info = pdf.metadata
                num_pages = len(pdf.pages)
                
                # Extract title or use filename
                title = info.title if info and info.title else os.path.basename(filepath)
                
                return {
                    "filename": os.path.basename(filepath),
                    "filepath": filepath,
                    "title": title,
                    "author": info.author if info and info.author else "Unknown",
                    "num_pages": num_pages,
                    "processed": False,
                    "has_tables": False,
                    "has_images": False,
                    "processed_pages": 0
                }
        except Exception as e:
            log.error(f"Error getting PDF metadata: {str(e)}")
            # Return basic info if metadata extraction fails
            return {
                "filename": os.path.basename(filepath),
                "filepath": filepath,
                "title": os.path.basename(filepath),
                "author": "Unknown",
                "num_pages": 0,
                "processed": False,
                "has_tables": False,
                "has_images": False,
                "processed_pages": 0
            }
    
    def add_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Add a new PDF file from uploaded content."""
        try:
            # Save the file
            filepath = os.path.join(self.pdf_dir, filename)
            with open(filepath, "wb") as f:
                f.write(file_content)
            
            # Get metadata
            metadata = self._get_pdf_metadata(filepath)
            self.loaded_pdfs[filename] = metadata
            
            log.info(f"Added new PDF: {filename} with {metadata['num_pages']} pages")
            return metadata
        except Exception as e:
            log.error(f"Error adding PDF {filename}: {str(e)}")
            raise
    
    def process_pdf(self, filename: str, max_pages: int = None) -> Dict[str, Any]:
        """Process a PDF file to extract text, tables, and images."""
        if filename not in self.loaded_pdfs:
            raise ValueError(f"PDF {filename} not found in loaded PDFs")
        
        filepath = self.loaded_pdfs[filename]["filepath"]
        metadata = self.loaded_pdfs[filename].copy()
        
        try:
            # Determine pages to process
            total_pages = metadata["num_pages"]
            pages_to_process = range(1, min(total_pages + 1, max_pages + 1) if max_pages else total_pages + 1)
            
            log.info(f"Processing PDF {filename}: {len(pages_to_process)} pages")
            
            # Process each page
            processed_data = {
                "text": {},
                "tables": {},
                "images": {},
                "sections": {}
            }
            
            # Extract text and structure using unstructured
            try:
                elements = partition_pdf(filepath, extract_images_in_pdf=True)
                
                # Process elements by type
                for i, element in enumerate(elements):
                    element_type = type(element).__name__
                    
                    if hasattr(element, "text") and element.text.strip():
                        page_num = element.metadata.page_number if hasattr(element.metadata, "page_number") else 0
                        
                        # Store text by page
                        if page_num not in processed_data["text"]:
                            processed_data["text"][page_num] = []
                        
                        # Add text with metadata
                        text_item = {
                            "text": element.text,
                            "type": element_type,
                            "page": page_num,
                            "metadata": element.metadata.__dict__ if hasattr(element, "metadata") else {}
                        }
                        
                        processed_data["text"][page_num].append(text_item)
                        
                        # If it's a section header, add to sections
                        if element_type == "Title" or element_type == "NarrativeText" and len(element.text) < 100 and element.text.isupper():
                            if page_num not in processed_data["sections"]:
                                processed_data["sections"][page_num] = []
                            
                            processed_data["sections"][page_num].append({
                                "title": element.text,
                                "page": page_num,
                                "index": i
                            })
                
                log.info(f"Extracted text from {len(processed_data['text'])} pages using unstructured")
            except Exception as e:
                log.error(f"Error extracting text with unstructured: {str(e)}")
                # Fallback to pdfplumber for text extraction
                self._extract_text_with_pdfplumber(filepath, processed_data, pages_to_process)
            
            # Extract tables using tabula and camelot
            has_tables = self._extract_tables(filepath, processed_data, pages_to_process)
            
            # Extract images
            has_images = self._extract_images(filepath, processed_data, pages_to_process)
            
            # Update metadata
            metadata["processed"] = True
            metadata["has_tables"] = has_tables
            metadata["has_images"] = has_images
            metadata["processed_pages"] = len(pages_to_process)
            metadata["processed_data"] = processed_data
            
            # Update stored metadata
            self.loaded_pdfs[filename] = metadata
            
            log.info(f"Completed processing PDF {filename}")
            return metadata
        except Exception as e:
            log.error(f"Error processing PDF {filename}: {str(e)}")
            raise
    
    def _extract_text_with_pdfplumber(self, filepath: str, processed_data: Dict, pages_to_process: range):
        """Extract text using pdfplumber as a fallback method."""
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_num in pages_to_process:
                    try:
                        page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing
                        text = page.extract_text()
                        
                        if text and text.strip():
                            if page_num not in processed_data["text"]:
                                processed_data["text"][page_num] = []
                            
                            processed_data["text"][page_num].append({
                                "text": text,
                                "type": "Text",
                                "page": page_num,
                                "metadata": {"source": "pdfplumber"}
                            })
                    except Exception as e:
                        log.error(f"Error extracting text from page {page_num}: {str(e)}")
            
            log.info(f"Extracted text from {len(processed_data['text'])} pages using pdfplumber")
        except Exception as e:
            log.error(f"Error in pdfplumber text extraction: {str(e)}")
    
    def _extract_tables(self, filepath: str, processed_data: Dict, pages_to_process: range) -> bool:
        """Extract tables from PDF using tabula and camelot."""
        has_tables = False
        
        # Try with tabula first (better for well-structured tables)
        try:
            tables = tabula.read_pdf(filepath, pages=','.join(map(str, pages_to_process)), multiple_tables=True)
            
            if tables:
                for i, table in enumerate(tables):
                    if not table.empty:
                        # Try to determine the page number
                        page_num = pages_to_process[0] + i if i < len(pages_to_process) else 0
                        
                        if page_num not in processed_data["tables"]:
                            processed_data["tables"][page_num] = []
                        
                        # Convert table to dict for storage
                        table_dict = {
                            "data": table.to_dict(orient="records"),
                            "columns": table.columns.tolist(),
                            "source": "tabula",
                            "page": page_num,
                            "index": i
                        }
                        
                        processed_data["tables"][page_num].append(table_dict)
                        has_tables = True
            
            log.info(f"Extracted {len(tables)} tables using tabula")
        except Exception as e:
            log.error(f"Error in tabula table extraction: {str(e)}")
        
        # Try with camelot for more complex tables
        try:
            for page_num in pages_to_process:
                tables = camelot.read_pdf(filepath, pages=str(page_num), flavor='lattice')
                
                if tables and len(tables) > 0:
                    if page_num not in processed_data["tables"]:
                        processed_data["tables"][page_num] = []
                    
                    for i, table in enumerate(tables):
                        if table.df.empty:
                            continue
                        
                        # Convert table to dict for storage
                        table_dict = {
                            "data": table.df.to_dict(orient="records"),
                            "columns": table.df.columns.tolist(),
                            "source": "camelot",
                            "page": page_num,
                            "index": i,
                            "accuracy": table.accuracy
                        }
                        
                        processed_data["tables"][page_num].append(table_dict)
                        has_tables = True
            
            log.info(f"Extracted tables from {len(processed_data['tables'])} pages using camelot")
        except Exception as e:
            log.error(f"Error in camelot table extraction: {str(e)}")
        
        return has_tables
    
    def _extract_images(self, filepath: str, processed_data: Dict, pages_to_process: range) -> bool:
        """Extract images from PDF pages."""
        has_images = False
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(filepath, first_page=min(pages_to_process), last_page=max(pages_to_process))
            
            for i, image in enumerate(images):
                page_num = min(pages_to_process) + i
                
                # Save the page image
                image_filename = f"{os.path.basename(filepath)}_{page_num}.png"
                image_path = os.path.join("data/images", image_filename)
                image.save(image_path)
                
                # Initialize images dict for this page
                if page_num not in processed_data["images"]:
                    processed_data["images"][page_num] = []
                
                # Add the full page image
                processed_data["images"][page_num].append({
                    "path": image_path,
                    "type": "full_page",
                    "page": page_num
                })
                
                has_images = True
                
                # Try to extract charts and figures using OpenCV
                try:
                    # Convert PIL image to OpenCV format
                    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Detect potential charts/figures (this is a simplified approach)
                    # A more sophisticated approach would use ML models for chart detection
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for j, contour in enumerate(contours):
                        # Filter small contours
                        if cv2.contourArea(contour) < 10000:  # Minimum area threshold
                            continue
                        
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Skip if too small or too large
                        if w < 100 or h < 100 or w > cv_image.shape[1] * 0.9 or h > cv_image.shape[0] * 0.9:
                            continue
                        
                        # Extract the region
                        region = cv_image[y:y+h, x:x+w]
                        
                        # Save the region as a potential chart/figure
                        region_filename = f"{os.path.basename(filepath)}_{page_num}_region_{j}.png"
                        region_path = os.path.join("data/images", region_filename)
                        cv2.imwrite(region_path, region)
                        
                        # Add to processed data
                        processed_data["images"][page_num].append({
                            "path": region_path,
                            "type": "region",
                            "page": page_num,
                            "coordinates": {"x": x, "y": y, "width": w, "height": h}
                        })
                except Exception as e:
                    log.error(f"Error extracting regions from page {page_num}: {str(e)}")
            
            log.info(f"Extracted images from {len(processed_data['images'])} pages")
        except Exception as e:
            log.error(f"Error in image extraction: {str(e)}")
        
        return has_images
    
    def get_pdf_list(self) -> List[Dict[str, Any]]:
        """Get a list of all loaded PDFs with metadata."""
        return [metadata for filename, metadata in self.loaded_pdfs.items()]
    
    def get_pdf_content(self, filename: str, content_type: str = "all", page: int = None) -> Dict[str, Any]:
        """Get specific content from a processed PDF."""
        if filename not in self.loaded_pdfs:
            raise ValueError(f"PDF {filename} not found in loaded PDFs")
        
        metadata = self.loaded_pdfs[filename]
        
        if not metadata.get("processed", False):
            raise ValueError(f"PDF {filename} has not been processed yet")
        
        processed_data = metadata.get("processed_data", {})
        
        if content_type == "all":
            if page:
                # Return all content types for a specific page
                return {
                    "text": processed_data.get("text", {}).get(page, []),
                    "tables": processed_data.get("tables", {}).get(page, []),
                    "images": processed_data.get("images", {}).get(page, []),
                    "sections": processed_data.get("sections", {}).get(page, [])
                }
            else:
                # Return all content for all pages
                return processed_data
        else:
            # Return specific content type
            if page:
                return processed_data.get(content_type, {}).get(page, [])
            else:
                return processed_data.get(content_type, {})
    
    def search_pdf_text(self, query: str, filename: str = None) -> List[Dict[str, Any]]:
        """Search for text in processed PDFs."""
        results = []
        
        # Determine which PDFs to search
        pdfs_to_search = [self.loaded_pdfs[filename]] if filename else self.loaded_pdfs.values()
        
        for pdf_metadata in pdfs_to_search:
            if not pdf_metadata.get("processed", False):
                continue
            
            processed_data = pdf_metadata.get("processed_data", {})
            text_data = processed_data.get("text", {})
            
            for page_num, page_texts in text_data.items():
                for text_item in page_texts:
                    text = text_item.get("text", "")
                    
                    if query.lower() in text.lower():
                        results.append({
                            "filename": pdf_metadata["filename"],
                            "title": pdf_metadata["title"],
                            "page": page_num,
                            "text": text,
                            "text_type": text_item.get("type", "Text"),
                            "context": self._get_text_context(text, query)
                        })
        
        return results
    
    def _get_text_context(self, text: str, query: str, context_size: int = 100) -> str:
        """Get context around a query match in text."""
        query_lower = query.lower()
        text_lower = text.lower()
        
        start_pos = text_lower.find(query_lower)
        if start_pos == -1:
            return ""
        
        # Get context around the match
        context_start = max(0, start_pos - context_size)
        context_end = min(len(text), start_pos + len(query) + context_size)
        
        # Extract context
        context = text[context_start:context_end]
        
        # Add ellipsis if needed
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context
