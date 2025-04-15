import os
import pandas as pd
import json
import io
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract
from typing import Optional, Dict, Any, List, Union
import streamlit as st

class FileHandler:
    """
    Handles all file operations including loading, processing, and validating files.
    Supports various file formats including CSV, Excel, JSON, and PDF.
    """
    def __init__(self, max_file_size_mb: int = 300):
        self.max_file_size_mb = max_file_size_mb
        self.supported_structured_data = ["xlsx", "xls", "csv", "json"]
        self.supported_documents = ["pdf"]
        self.supported_file_types = self.supported_structured_data + self.supported_documents
    
    def validate_file(self, file) -> bool:
        """
        Validates if the file is of supported type and within size limits.
        Raises ValueError if validation fails.
        """
        if file.size > self.max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File size exceeds {self.max_file_size_mb}MB limit")
        
        extension = os.path.splitext(file.name)[1].lower()[1:]
        if extension not in self.supported_file_types:
            raise ValueError(f"Unsupported file type. Please upload {', '.join(self.supported_file_types)}")
        
        return True
    
    def get_dataframe_from_file(self, file) -> pd.DataFrame:
        """
        Load a DataFrame from the uploaded file.
        Supports CSV, JSON, and Excel formats.
        """
        file_extension = os.path.splitext(file.name)[1].lower()
        file.seek(0)  # Reset file pointer
        
        if file_extension == ".csv":
            return pd.read_csv(file)
        elif file_extension == ".json":
            try:
                return pd.read_json(file, orient='records')
            except:
                try:
                    return pd.read_json(file, orient='split')
                except:
                    json_data = json.load(file)
                    if isinstance(json_data, dict):
                        return pd.DataFrame([json_data])
                    elif isinstance(json_data, list):
                        return pd.DataFrame(json_data)
                    else:
                        raise ValueError("Unsupported JSON structure")
        else:  # Excel files
            return pd.read_excel(file)
    
    def process_pdf(self, file) -> Optional[str]:
        """
        Process a PDF file and extract its text content.
        Uses PyMuPDF for text extraction and falls back to OCR if needed.
        """
        try:
            # Read PDF file bytes
            pdf_bytes = file.read()
            
            # Try PyMuPDF first for text extraction
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_content = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_content.append(page.get_text())
            
            full_text = "\n".join(text_content)
            
            # If text extraction yields empty or minimal results, try OCR
            if len(full_text.strip()) < 100:  # Arbitrary threshold for minimum text
                # Convert PDF to images
                images = convert_from_bytes(pdf_bytes)
                ocr_text = []
                
                for img in images:
                    # Perform OCR on each page
                    page_text = pytesseract.image_to_string(img)
                    ocr_text.append(page_text)
                
                full_text = "\n".join(ocr_text)
            
            doc.close()
            return full_text
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    
    def read_file_to_text(self, file) -> Optional[str]:
        """
        Enhanced file reading function with PDF support.
        Returns a formatted text representation of the file content.
        """
        file_extension = os.path.splitext(file.name)[1].lower()
        try:
            if file_extension == ".pdf":
                text_content = self.process_pdf(file)
                if text_content:
                    return f"Document Content:\n\n{text_content}"
                return None
            else:
                # For structured data files
                df = self.get_dataframe_from_file(file)
                
                # Import here to avoid circular imports
                from visualization_utils import get_data_summary
                
                # Get data summary for structured data
                data_summary = get_data_summary(df)
                buffer = io.StringIO()
                df.info(buf=buffer)
                df_info = buffer.getvalue()
                df_sample = df.head(20).to_string()
                
                return f"Data Summary:\n{data_summary}\n\nDataFrame Info:\n{df_info}\n\nSample Data (first 20 rows):\n{df_sample}"
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None