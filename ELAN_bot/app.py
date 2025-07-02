"""
Main application file for ELAN-Bot.
"""

import os
import sys
import warnings

# Set environment variable for CPU-only torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Suppress warnings for cleaner startup
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from services import ElanAssistant
from ui import GradioInterface


class ElanBotApp:
    """Main application class for ELAN-Bot."""
    
    def __init__(self):
        """Initialize the ELAN-Bot application."""
        self.assistant = ElanAssistant()
        self.interface = GradioInterface(self.chat_handler)
    
    def chat_handler(self, message: str, history) -> str:
        """
        Handle chat messages from Gradio interface.
        
        Args:
            message: User's input message
            history: Chat history (unused but required by Gradio)
            
        Returns:
            str: Assistant's response
        """
        try:
            return self.assistant.process_message(message)
        except Exception as e:
            print(f"Error in chat handler: {e}")
            return "I'm sorry, an error occurred while processing your request."
    
    def launch(self, share: bool = False, **kwargs):
        """
        Launch the application.
        
        Args:
            share: Whether to create a public link (False for HF Spaces)
            **kwargs: Additional arguments for Gradio launch
        """
        # For Hugging Face Spaces, we don't need share=True
        self.interface.launch(share=share, **kwargs)


def main():
    """Main entry point for the application."""
    app = ElanBotApp()
    app.launch()


if __name__ == "__main__":
    main()