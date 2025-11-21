from typing import Dict, List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import os

# Define the structured output
class MeetingOutput(BaseModel):
    summary: str
    decisions: List[Dict]        
    action_items: List[Dict]

# Initialize Ollama LLM (Phi-3 Mini)
llm = OllamaLLM(model="phi3")

# Create a PromptTemplate
template = """
You are a meeting assistant.
Given the following meeting transcript, extract:

1. A concise summary of the meeting.
2. Decisions made.
3. Action items with speaker names.

Return the output in JSON format with keys: summary, decisions, action_items.

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
print("Summary:\n", output.summary)
print("\nDecisions:\n", output.decisions)
print("\nAction Items:\n", output.action_items)

























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


