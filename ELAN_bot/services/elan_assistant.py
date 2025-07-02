"""
Main ELAN assistant service that coordinates all components.
"""

from .vector_search import VectorSearchService
from .llm_service import LLMService
from utils.text_processing import TextProcessor


class ElanAssistant:
    """Main assistant service that coordinates all components."""
    
    def __init__(self):
        """Initialize the ELAN assistant with all required services."""
        self.vector_search = VectorSearchService()
        self.llm_service = LLMService()
        self.text_processor = TextProcessor()
    
    def process_message(self, message: str) -> str:
        """
        Process a user message and return appropriate response.
        
        Args:
            message: The user's input message
            
        Returns:
            str: Generated response
        """
        # Check if message contains XML/EAF content
        if self.text_processor.is_xml_content(message):
            return self._process_xml_modification(message)
        else:
            return self._process_question(message)
    
    def _process_question(self, question: str) -> str:
        """
        Process a regular question using vector search and LLM.
        
        Args:
            question: The user's question
            
        Returns:
            str: Generated answer
        """
        # Get relevant context from vector search
        context = self.vector_search.get_context(question)
        
        # Generate answer using LLM
        response = self.llm_service.generate_answer(question, context)
        
        return response
    
    def _process_xml_modification(self, eaf_content: str) -> str:
        """
        Process XML/EAF file modification request.
        
        Args:
            eaf_content: The EAF file content with instructions
            
        Returns:
            str: Modified EAF content
        """
        try:
            # Split content into instructions and chunks
            instructions, chunks = self.text_processor.split_eaf_content(eaf_content)
            
            # Process each chunk
            processed_chunks = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks, 1):
                processed_chunk = self.llm_service.process_xml_chunk(
                    chunk=chunk,
                    instructions=instructions,
                    current_chunk=i,
                    total_chunks=total_chunks
                )
                processed_chunks.append(processed_chunk)
            
            # Combine processed chunks
            combined_result = self.text_processor.combine_chunks(processed_chunks)
            
            return combined_result
            
        except Exception as e:
            print(f"Error in EAF file modification: {e}")
            return "I'm sorry, an error occurred while modifying the EAF file."