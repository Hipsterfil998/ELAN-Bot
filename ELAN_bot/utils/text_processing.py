"""
Text processing utilities for ELAN-Bot application.
"""

import tiktoken
from typing import List, Tuple
from config.settings import DEFAULT_TOKENIZER_MODEL, CHUNK_SIZE


class TextProcessor:
    """Utility class for text processing operations."""
    
    def __init__(self, model: str = DEFAULT_TOKENIZER_MODEL):
        """
        Initialize the text processor.
        
        Args:
            model: The tokenizer model to use
        """
        self.model = model
        self.tokenizer = None
    
    def _get_tokenizer(self):
        """Get or create the tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        return self.tokenizer
    
    def split_eaf_content(
        self, 
        eaf_file: str, 
        chunk_size: int = CHUNK_SIZE
    ) -> Tuple[str, List[str]]:
        """
        Split EAF file content into smaller chunks based on token count.
        
        Args:
            eaf_file: The complete EAF file content
            chunk_size: Maximum number of tokens per chunk
            
        Returns:
            Tuple containing (instructions, text_chunks) where:
            - instructions: Text before the XML content
            - text_chunks: List of XML chunks split by token count
        """
        # Separate initial instructions from XML content
        instructions = ""
        xml_start = eaf_file.find("<?xml")
        
        if xml_start > 0:
            instructions = eaf_file[:xml_start].strip()
            eaf_content = eaf_file[xml_start:]
        else:
            eaf_content = eaf_file
        
        # Tokenize the content
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(eaf_content)
        
        # Split tokens into chunks
        token_chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i+chunk_size]
            token_chunks.append(chunk)
        
        # Decode chunks back to text
        text_chunks = []
        for chunk in token_chunks:
            chunk_text = tokenizer.decode(chunk)
            text_chunks.append(chunk_text)
        
        return instructions, text_chunks
    
    @staticmethod
    def combine_chunks(processed_chunks: List[str]) -> str:
        """
        Combine processed chunks into a single string.
        
        Args:
            processed_chunks: List of processed chunk strings
            
        Returns:
            str: Combined content
        """
        return "".join(processed_chunks)
    
    @staticmethod
    def is_xml_content(message: str) -> bool:
        """
        Check if the message contains XML/EAF content.
        
        Args:
            message: The message to check
            
        Returns:
            bool: True if message contains XML content
        """
        xml_indicators = ["<?xml", "<eaf", "<ANNOTATION"]
        return any(indicator in message for indicator in xml_indicators)