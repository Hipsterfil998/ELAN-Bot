import gradio as gr
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
import time
import asyncio


# Configure the inference client
def get_inference_client():
    return InferenceClient(
        provider="hf-inference",
        api_key=os.environ.get("HF_TOKEN", "")
    )

# Function to get context from vector searches
def get_context(query, encoder_model="nomic-ai/nomic-embed-text-v1.5"):
    try:
        # Get client path from environment variable or use default
        client_path = "qdrant_data"
        
        # Load the encoder
        encoder = SentenceTransformer(encoder_model, trust_remote_code=True)
        
        # Initialize the Qdrant client
        client = QdrantClient(path=client_path)
        
        # Encode the query
        query_vector = encoder.encode(query).tolist()
        
        # Execute the search
        hits = client.query_points(
            collection_name="ELAN_docs_pages",
            query=query_vector,
            limit=3,
        ).points
            
        # Get the context content
        context = "\n".join([hit.payload['content'] for hit in hits])
        
        return context
    except Exception as e:
        print(f"Error in vector search: {e}")
        return "I'm sorry, it was not possible to find any relevant information."

# Function to generate a response based on context
def get_answer(query, context, model="meta-llama/Llama-3.3-70B-Instruct"):
    try:
        client = get_inference_client()
        
        PROMPT = """<|start_header_id|>user<|end_header_id|>
                    Use exclusively the information contained in the provided context to reformulate the text in about 120 words.
                    take into consideration the provided question as a reference for the formulation of the answer.
                    To be more clear and coincise use numbered lists when giving instructions.
                    Make sure the reformulation maintains the original meaning.
                    In the output, check that there are no grammatical errors. If you find errors, correct them.
                    Do not add information that is not present in the original text.
                    The output must have the same language of the question. If not translate it.
                    In the output, never say that you are summarizing the text and never mention the ELAN manual and its chapthers. In this latter case tell to be more specific with the question.
                    
                    Context: {contesto}, question: {domanda} <|eot_id|>"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a virtual assistant that helps the user in using an annotation software called ELAN. Detect the question language and translate the output in the same language if it is not English. Your task is to summarize information and guide the user in the usage of the software. <|eot_id|>"},
                {"role": "user", "content": PROMPT.format(contesto=context, domanda=query)},
                {"role": "assistant", "content": "Here is what you're looking for: "}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in response generation: {e}")
        return "I'm sorry, an error occurred while generating the response."

# Function to modify XML code
def modify_xml(eaf_file, model="meta-llama/Llama-3.3-70B-Instruct"):
    try:
        client = get_inference_client()
        
        PROMPT = """<|start_header_id|>user<|end_header_id|>
                    
                    Example eaf file:
                    
                    <?xml version="1.0" encoding="UTF-8"?>
                    <ANNOTATION_DOCUMENT AUTHOR="Giulia Bianchi" DATE="2025-04-08T14:30:00+01:00" FORMAT="3.0" VERSION="3.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
                        <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
                            <MEDIA_DESCRIPTOR MEDIA_URL="file:///C:/Progetti/multilingual_corpus/video01.mp4" MIME_TYPE="video/mp4" RELATIVE_MEDIA_URL="./video01.mp4"/>
                            <PROPERTY NAME="lastUsedAnnotationId">15</PROPERTY>
                        </HEADER>
                        <TIME_ORDER>
                            <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="2500"/>
                            <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="5200"/>
                            <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="2500"/>
                            <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="5200"/>
                            <TIME_SLOT TIME_SLOT_ID="ts5" TIME_VALUE="2500"/>
                            <TIME_SLOT TIME_SLOT_ID="ts6" TIME_VALUE="4100"/>
                        </TIME_ORDER>
                        <TIER LINGUISTIC_TYPE_REF="utterance" PARTICIPANT="Speaker1" TIER_ID="Italiano">
                            <ANNOTATION>
                                <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                                    <ANNOTATION_VALUE>Buongiorno, come sta oggi?</ANNOTATION_VALUE>
                                </ALIGNABLE_ANNOTATION>
                            </ANNOTATION>
                        </TIER>
                        <TIER LINGUISTIC_TYPE_REF="translation" PARENT_REF="Italiano" PARTICIPANT="Translator" TIER_ID="English">
                            <ANNOTATION>
                                <REF_ANNOTATION ANNOTATION_ID="a2" ANNOTATION_REF="a1">
                                    <ANNOTATION_VALUE>Good morning, how are you today?</ANNOTATION_VALUE>
                                </REF_ANNOTATION>
                            </ANNOTATION>
                        </TIER>
                        <TIER LINGUISTIC_TYPE_REF="gesture" PARTICIPANT="Speaker1" TIER_ID="Gestures">
                            <ANNOTATION>
                                <ALIGNABLE_ANNOTATION ANNOTATION_ID="a3" TIME_SLOT_REF1="ts5" TIME_SLOT_REF2="ts6">
                                    <ANNOTATION_VALUE>Inclinazione della testa mentre saluta</ANNOTATION_VALUE>
                                </ALIGNABLE_ANNOTATION>
                            </ANNOTATION>
                        </TIER>
                        <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="utterance" TIME_ALIGNABLE="true"/>
                        <LINGUISTIC_TYPE CONSTRAINTS="Symbolic_Association" GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="translation" TIME_ALIGNABLE="false"/>
                        <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="gesture" TIME_ALIGNABLE="true"/>
                        <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
                        <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered" STEREOTYPE="Symbolic_Subdivision"/>
                        <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
                        <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
                    </ANNOTATION_DOCUMENT>
         
                    Modify the provided eaf file according to the instructions given by the user.
                    Take the above example eaf file to better understand the file structure and where instructions start. 
                    Parse the eaf file step by step. Remember that is XML-based.
                    Then follow the instructions step by step.
                    Don't add any additional information, explanations or reasoning steps to the ouput if not explicitely requested in the instructions.
                    Report the final output only.
                    
                    Provided .eaf file and instructions: {code} <|eot_id|>"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                "You are a virtual assistant that helps the user in using an annotation software called ELAN."
                "An annotation file (eaf) is the document that contains all the information about tiers (their attributes and dependency relations), annotations, and time alignments and links to media files."
                "Your task is to modify the given eaf file and extract information strictly following the instructions given by the user.<|eot_id|>"},
                {"role": "user", "content": PROMPT.format(code=eaf_file)},
                {"role": "assistant", "content": "Here is your output: "}
            ],
            temperature=0.7,
            max_tokens=None
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in eaf file modification: {e}")
        return "I'm sorry, an error occurred while modifying the eaf file."

# Stream response in chatbot
def elan_assistant(message, history):
    if "<?xml" in message or "<eaf" in message or "<ANNOTATION" in message:
        response = modify_xml(message)
        return response
    else:
        context = get_context(message)
        full_response = get_answer(message, context)
        return full_response
    

# Build the Gradio app with streaming enabled
demo = gr.ChatInterface(
    fn=elan_assistant,
    title="ELAN-Bot 🤖 ",
    description="""Hello there!👋\nI'm ELAN-Bot, a virtual assistant designed to help you with the ELAN annotation software. You can ask me questions about:\n
    - 📚 software usage: how to use ELAN and its main features
    - 💻 XML code: modify the EAF file by providing me with the copy-pasted XML code and some instructions (e.g --> instructions: [.eaf file] extract ... from this .eaf file. Then report me the result as text.)\n
    Software usage functionality available in English 🇬🇧, Spanish 🇪🇸, Italian 🇮🇹, French 🇫🇷 and German 🇩🇪
    Based on Llama 3.3 70B""",
    examples=[
        "How can I add a new tier in ELAN?",
        "¿Cómo puedo exportar anotaciones en formato txt? ",
        "Come posso cercare all'interno delle annotazioni?"
    ],
    theme=gr.themes.Soft(),
    type="messages",
    textbox=gr.Textbox(placeholder="Ask me anything about ELAN..."),
    autoscroll = False,
    show_progress = 'full'
)

# App startup
if __name__ == "__main__":
    # Enable built-in Gradio streaming
    demo.queue()
    demo.launch(share=True)
