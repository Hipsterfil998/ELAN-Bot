"""
Prompt templates package for ELAN-Bot application.
"""

from .system_prompts import ANSWER_SYSTEM_PROMPT, XML_SYSTEM_PROMPT
from .user_prompts import ANSWER_USER_PROMPT, XML_USER_PROMPT
from .assistant_prompts import ANSWER_ASSISTANT_PROMPT, XML_ASSISTANT_PROMPT

__all__ = [
    "ANSWER_SYSTEM_PROMPT",
    "XML_SYSTEM_PROMPT", 
    "ANSWER_USER_PROMPT",
    "XML_USER_PROMPT",
    "ANSWER_ASSISTANT_PROMPT",
    "XML_ASSISTANT_PROMPT"
]