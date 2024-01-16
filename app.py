from fastapi import FastAPI, UploadFile, File
import uvicorn
import json

from clarifai.client.model import Model
from clarifai.client.input import Inputs
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

def extract_json(text):
    return text.split('```json')[1].split('```')[0]

def inference(model_url, inputs, params={}):
    prediction = Model(model_url).predict(inputs=[inputs], inference_params=params)
    return prediction.outputs[0].data

@app.post("/process-feelings")
async def process_feelings(input: str):
    prompt = f'''
    Your task is to score a happiness scale out of 10 from the user feelings. 
    If the response isn't satisfactory enough, please ask a follow-up question. 
    Parse your output in json format with two keys: "happiness_scale" and "follow_up_question". 
    Write empty values as null. 
    user response: {input}
    '''
    model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    inputs = Inputs.get_text_input(input_id="", raw_text=prompt)
    response = inference(model_url, inputs, {'temperature': 0})
    data = json.loads(extract_json(response.text.raw))
    return data


# @app.post("/create-art")
# async def create_art(description: str):
#     # Convert description to image
#     # Placeholder for image generation logic
#     return {"message": "Image generated from description"}

# @app.post("/text-to-speech")
# async def text_to_speech(text: str):
#     # Convert text to speech
#     # Placeholder for text-to-speech logic
#     return {"message": "Speech generated from text"}

# @app.post("/image-to-text")
# async def image_to_text(image: UploadFile = File(...)):
#     # Extract text from image
#     # Placeholder for image-to-text extraction logic
#     return {"extracted_text": "Extracted text from image"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
