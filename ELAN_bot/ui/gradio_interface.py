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
        Initialize and configure the Gradio chat interface.

        Args:
            chat_handler: Function to handle chat messages
        """
        self.demo = gr.ChatInterface(
            fn=chat_handler,
            title=APP_TITLE,
            description=APP_DESCRIPTION,
            examples=APP_EXAMPLES,
            theme=gr.themes.Soft(),
            type="messages",
            textbox=gr.Textbox(placeholder=TEXTBOX_PLACEHOLDER),
            cache_examples=False,
        )

    def launch(self, share: bool = False, **kwargs):
        """
        Launch the Gradio application.

        Args:
            share: Whether to create a public link (False for HF Spaces)
            **kwargs: Additional arguments for Gradio launch
        """
        self.demo.queue()
        self.demo.launch(share=share, **kwargs)
