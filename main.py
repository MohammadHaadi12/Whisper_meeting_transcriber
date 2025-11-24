from typing import Dict, List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import os
from db import init_db, save_meeting, load_meeting_transcript

init_db()

# Define the structured output
class MeetingOutput(BaseModel):
    title: str
    summary: str
    decisions: List[Dict]        
    action_items: List[Dict]

# Initialize Ollama LLM (Phi-3 Mini)
llm = OllamaLLM(model="phi3")

# Create a PromptTemplate
template = """
You are a strict JSON-generating meeting assistant.

Your job is to extract structured information from a transcript.

IMPORTANT — You must follow these rules:
- OUTPUT ONLY VALID JSON
- DO NOT include commentary, notes, markdown, or explanations
- KEYS MUST MATCH EXACTLY:
  - "title": string
  - "summary": string
  - "decisions": list of objects
  - "action_items": list of objects
- "decisions" MUST ALWAYS be a LIST, even if there is only one decision
- "action_items" MUST ALWAYS be a LIST
- DO NOT add extra fields
- DO NOT reorder keys

Here is the REQUIRED JSON structure you MUST follow:

{{
  "title": "string",
  "summary": "string",
  "decisions": [
       {{"decision_description": "string"}}
  ],
  "action_items": [
       {{"speaker": "string", "item": "string"}}
  ]
}}

Now extract the information strictly following this format.

Transcript:
{text}
"""


prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)


# Create parser for structured output
parser = PydanticOutputParser(pydantic_object=MeetingOutput)

# Combine into a single chain
chain = prompt | llm | parser

# Load your transcript
with open("transcript.txt", "r", encoding="utf-8") as f:
    transcript_text = f.read()

# Run the chain
output = chain.invoke({"text": transcript_text})

# Access structured fields
print("Title:\n", output.title)
print("Summary:\n", output.summary)
print("\nDecisions:\n", output.decisions)
print("\nAction Items:\n", output.action_items)



save_meeting(output.title, transcript_text)

print("Meeting is saved to the database")



















# from openai import OpenAI
# import base64
# import os
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("OPENAI_KEY")

# client = OpenAI(api_key = api_key)

# # def to_data_url(path):
# #     with open(path, "rb") as f:
# #         return "data:audio/wav;base64," + base64.b64encode(f.read()).decode()

# # with open("testing.mp3", "rb") as audio_file:
# #     result = client.audio.transcriptions.create(
# #         model="gpt-4o-transcribe-diarize",
# #         file=audio_file,
# #         response_format="diarized_json",
# #         chunking_strategy="auto",
# #     )

# # with open("transcript.txt", "w", encoding="utf-8") as f:
# #     for segment in result.segments:
# #         f.write(f"{segment.speaker}: {segment.text}\n")



# with open("transcript.txt", "r", encoding="utf-8") as f:
#     transcript_text = f.read()

# print(transcript_text)


