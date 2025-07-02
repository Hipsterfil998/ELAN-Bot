"""
Gradio interface components for ELAN-Bot application.
"""

import gradio as gr
from config.settings import (
    APP_TITLE, APP_DESCRIPTION, APP_EXAMPLES, TEXTBOX_PLACEHOLDER
)


class GradioInterface:
    """Class for handling Gradio interface setup and configuration."""
    
    def __init__(self, chat_handler):
        """
        Initialize the Gradio interface.
        
        Args:
            chat_handler: Function to handle chat messages
        """
        self.chat_handler = chat_handler
        self.demo = None
    
    def create_interface(self) -> gr.ChatInterface:
        """
        Create and configure the Gradio chat interface.
        
        Returns:
            gr.ChatInterface: Configured Gradio interface
        """
        self.demo = gr.ChatInterface(
            fn=self.chat_handler,
            title=APP_TITLE,
            description=APP_DESCRIPTION,
            examples=APP_EXAMPLES,
            theme=gr.themes.Soft(),
            type="messages",
            textbox=gr.Textbox(placeholder=TEXTBOX_PLACEHOLDER),
            cache_examples=False,  # Disable example caching to avoid format issues
        )
        
        return self.demo
    
    def launch(self, share: bool = False, **kwargs):
        """
        Launch the Gradio application.
        
        Args:
            share: Whether to create a public link (False for HF Spaces)
            **kwargs: Additional arguments for Gradio launch
        """
        if self.demo is None:
            self.create_interface()
        
        # Enable queueing for better performance
        self.demo.queue()
        self.demo.launch(share=share, **kwargs)