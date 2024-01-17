from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
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
async def process_feelings(conversation: str):
    prompt = f'''
    Your task is to score a mood scale out of 10 from the user feelings. (10 is best mood, 1 is worst mood)
    If the response isn't satisfactory enough, please ask a follow-up question. 
    The follow-up question is meant to understand the user feelings better, avoid writing suggestions and don't ask over-detailed questions.
    Parse your output in json format with two keys: "mood_scale" and "follow_up_question". 
    Write empty values as null. 
    {conversation}
    '''
    model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    inputs = Inputs.get_text_input(input_id="", raw_text=prompt)
    response = inference(model_url, inputs, {'temperature': 0})
    data = json.loads(extract_json(response.text.raw))
    return data

@app.post("/generate-prompts")
async def generate_prompts(mood: int, conversation: str):
    prompt = f'''
    The user's mood scale is: {mood}. (10 is best mood, 1 is worst mood)
    Analyze the following conversation:
    {conversation}
    Your task is to write suggestions for art drawing, journaling and meditation. 
    Each suggestion should just be one short sentence. 
    Furthermore, please write 2 daily affirmations.
    Parse your output in json format with these keys: "art", "journal", "meditation", "affirmation".
    '''
    model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    inputs = Inputs.get_text_input(input_id="", raw_text=prompt)
    response = inference(model_url, inputs, {'temperature': 0})
    data = json.loads(extract_json(response.text.raw))
    return data

@app.post("/describe-art/")
async def describe_art(file: UploadFile = File(...)):
    image = await file.read()
    model_url = "https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision"
    prompt = "Infer the person's feelings from the given drawing"
    inputs = Inputs.get_multimodal_input(input_id="", image_bytes=image, raw_text=prompt)
    data = inference(model_url, inputs, {'temperature': 0})
    return {'meaning': data.text.raw}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
