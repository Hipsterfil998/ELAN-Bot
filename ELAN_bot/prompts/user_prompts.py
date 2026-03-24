"""
User prompts for ELAN-Bot application.
"""

ANSWER_USER_PROMPT = """Use exclusively the information contained in the provided context to reformulate the text in about 120 words.
Take into consideration the provided question as a reference for the formulation of the answer.
To be more clear and concise use numbered lists when giving instructions.
Make sure the reformulation maintains the original meaning.
In the output, check that there are no grammatical errors. If you find errors, correct them.
Do not add information that is not present in the original text.
The output must have the same language of the question. If not translate it.
In the output, never say that you are summarizing the text and never mention the ELAN manual and its chapters. In this latter case tell to be more specific with the question.

Context: {context}, question: {question}"""

XML_USER_PROMPT = """## EAF File Structure Reference (Detailed)

### Root Element
- ANNOTATION_DOCUMENT
  - Attributes: AUTHOR, DATE, FORMAT, VERSION, xmlns:xsi, xsi:noNamespaceSchemaLocation
  - Purpose: Contains the entire annotation document and its metadata

### Header Section
- HEADER
  - Attributes: MEDIA_FILE, TIME_UNITS
  - Child elements:
    - MEDIA_DESCRIPTOR (Attributes: MEDIA_URL, MIME_TYPE, RELATIVE_MEDIA_URL)
    - PROPERTY (Attributes: NAME, with values like "URN", "lastUsedAnnotationId")
  - Purpose: Specifies media references and time measurement units

### Time Structure
- TIME_ORDER
  - Child elements: TIME_SLOT (multiple)
    - TIME_SLOT attributes: TIME_SLOT_ID, TIME_VALUE
  - Purpose: Defines timestamps for aligning annotations with media

### Annotation Tiers
- TIER (multiple)
  - Attributes: LINGUISTIC_TYPE_REF, TIER_ID, PARTICIPANT, DEFAULT_LOCALE, LANG_REF
  - Child elements: ANNOTATION (multiple)
    - Types:
      - ALIGNABLE_ANNOTATION
        - Attributes: ANNOTATION_ID, TIME_SLOT_REF1, TIME_SLOT_REF2
        - Child element: ANNOTATION_VALUE (contains transcribed text)
      - REF_ANNOTATION
        - Attributes: ANNOTATION_ID, ANNOTATION_REF, PREVIOUS_ANNOTATION
        - Child element: ANNOTATION_VALUE (contains token or tag value)
  - Purpose: Organizes annotations by participant or type

### Token and POS Tagging Tiers
- S1_token, S2_token (Token tiers)
  - Contains: REF_ANNOTATION elements
    - Each references a parent annotation via ANNOTATION_REF
    - ANNOTATION_VALUE contains individual words
  - Purpose: Breaks utterances into individual tokens

- S1_POStags, S2_POStags (Part-of-speech tag tiers)
  - Contains: REF_ANNOTATION elements
    - Each references a token via ANNOTATION_REF
    - Attributes: CVE_REF (reference to controlled vocabulary entry)
    - ANNOTATION_VALUE contains grammatical tag code
  - Purpose: Provides grammatical information for each token

### Linguistic Type Definitions
- LINGUISTIC_TYPE (multiple)
  - Attributes: LINGUISTIC_TYPE_ID, CONSTRAINTS, CONTROLLED_VOCABULARY_REF, GRAPHIC_REFERENCES, TIME_ALIGNABLE
  - Purpose: Defines types of annotations and their constraints

### Language and Locale Settings
- LOCALE
  - Attributes: COUNTRY_CODE, LANGUAGE_CODE
  - Purpose: Specifies language region

- LANGUAGE
  - Attributes: LANG_ID, LANG_LABEL, LANG_DEF
  - Purpose: Defines language parameters

### Constraints
- CONSTRAINT (multiple)
  - Attributes: DESCRIPTION, STEREOTYPE
  - Purpose: Defines relationships between annotations

### Controlled Vocabulary
- CONTROLLED_VOCABULARY
  - Attributes: CV_ID
  - Child elements:
    - DESCRIPTION (Attributes: LANG_REF)
    - CV_ENTRY_ML (multiple)
      - Attributes: CVE_ID
      - Child elements: CVE_VALUE
      - Attributes: DESCRIPTION, LANG_REF
    - Purpose: Provides standard annotation values for consistent tagging

## Processing Instructions
1. Parse the XML chunk provided below step by step. Refer to the structure above to understand where information is located.
2. Remember that this is chunk {current_chunk} of {total_chunks}. 
3. Apply the modification requested by the user's instructions. If there's no element to modify just let the chunk as it is.
4. Return the XML content for this chunk.

## Output Requirements
- file format: EAF file
- Maintain proper XML formatting and indentation
- Do not include explanations, commentary, or reasoning in the output
user instructions: {instructions}

EAF file chunk: {chunk}"""