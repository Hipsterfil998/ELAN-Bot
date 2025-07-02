"""
LLM service for ELAN-Bot application.
"""

from huggingface_hub import InferenceClient
from config.settings import HF_TOKEN, DEFAULT_LLM_MODEL, TEMPERATURE, ANSWER_MAX_TOKENS, MAX_TOKENS
from prompts import (
    ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT, ANSWER_ASSISTANT_PROMPT,
    XML_SYSTEM_PROMPT, XML_USER_PROMPT, XML_ASSISTANT_PROMPT
)


class LLMService:
    """Service for handling LLM interactions."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.client = None
    
    def _get_client(self) -> InferenceClient:
        """Get or create the inference client."""
        if self.client is None:
            if not HF_TOKEN:
                raise ValueError("HF_TOKEN environment variable is required but not set")
            # Updated initialization for new huggingface_hub version
            self.client = InferenceClient(
                token=HF_TOKEN
            )
        return self.client
    
    def generate_answer(
        self, 
        query: str, 
        context: str, 
        model: str = DEFAULT_LLM_MODEL
    ) -> str:
        """
        Generate an answer based on query and context.
        
        Args:
            query: The user's question
            context: The relevant context from vector search
            model: The LLM model to use
            
        Returns:
            str: Generated answer
        """
        try:
            client = self._get_client()
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {
                        "role": "user", 
                        "content": ANSWER_USER_PROMPT.format(
                            context=context, 
                            question=query
                        )
                    },
                    {"role": "assistant", "content": ANSWER_ASSISTANT_PROMPT}
                ],
                temperature=TEMPERATURE,
                max_tokens=ANSWER_MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in response generation: {e}")
            return "I'm sorry, an error occurred while generating the response."
    
    def process_xml_chunk(
        self, 
        chunk: str, 
        instructions: str, 
        current_chunk: int, 
        total_chunks: int,
        model: str = DEFAULT_LLM_MODEL
    ) -> str:
        """
        Process a single XML chunk with given instructions.
        
        Args:
            chunk: The XML chunk to process
            instructions: User instructions for modification
            current_chunk: Current chunk number
            total_chunks: Total number of chunks
            model: The LLM model to use
            
        Returns:
            str: Processed XML chunk
        """
        try:
            client = self._get_client()
            
            chunk_prompt = XML_USER_PROMPT.format(
                current_chunk=current_chunk,
                total_chunks=total_chunks,
                chunk=chunk,
                instructions=instructions
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": XML_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk_prompt},
                    {"role": "assistant", "content": XML_ASSISTANT_PROMPT}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error processing XML chunk {current_chunk}: {e}")
            return chunk  # Return original chunk if processing fails