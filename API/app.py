from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import uvicorn
import json
import io
import os

from clarifai.client.model import Model
from clarifai.client.input import Inputs

import langchain
from langchain_community.llms import Clarifai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import pinecone


langchain.debug = True
load_dotenv()
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment='gcp-starter')
vector_store = Pinecone.from_existing_index('meditation', OpenAIEmbeddings())

MODEL_URL="https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
llm = Clarifai(model_url=MODEL_URL)

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=vector_store.as_retriever())

with open('meditation_prompt.txt', 'r') as f:
    template = f.read()
prompt = PromptTemplate(
    input_variables = ["user_name", "query", "user_age",
                        "user_gender", "user_struggle",
                         "user_emotion", "meditation_length"],
    template=template
)


app = FastAPI()

def extract_json(text):
    return text.split('```json')[1].split('```')[0]

def inference(model_url, inputs, params={}):
    prediction = Model(model_url).predict(inputs=[inputs], inference_params=params)
    return prediction.outputs[0].data

@app.post("/process-feelings")
async def process_feelings(conversation: str):
    prompt = f'''
    You are a happiness scorer. Score the happiness of {conversation} from 1 to 10, 1 being not happy at all to 10 being extremely happy. 
    If you cannot score happiness, ask a noninvasive question to get a better description. If you are able to give a score, then only provide the score out of 10. 
    '''
    model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    inputs = Inputs.get_text_input(input_id="", raw_text=prompt)
    response = inference(model_url, inputs, {'temperature': 0})
    data = json.loads(extract_json(response.text.raw))
    return data

@app.post("/generate-prompts")
async def generate_prompts(mood: int, conversation: str):
    prompt = f'''
    The user's happiness scale is: {mood}. 1 being not happy at all to 10 being extremely happy
    Analyze the following conversation:
    {conversation}
    Based on the {mood} and {conversation}: Return a journaling prompt and a short art prompt to help process with emotional processing.
    Also, provide 2 daily affirmations based on the {mood} and {conversation}
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
    prompt = "Describe what the image could be trying to communicate about their emotional state in 1-2 sentences."
    inputs = Inputs.get_multimodal_input(input_id="", image_bytes=image, raw_text=prompt)
    data = inference(model_url, inputs, {'temperature': 0})
    return {'meaning': data.text.raw}

@app.post("/text-to-speech/")
async def text_to_speech(text: str):
    model_url = 'https://clarifai.com/openai/tts/models/openai-tts-1'
    inputs = Inputs.get_text_input(input_id="", raw_text=text)
    data = inference(model_url, inputs, {'voice': 'shimmer', 'speed': 1})

    audio_buffer = io.BytesIO(data.audio.base64)
    audio_buffer.seek(0)
    return StreamingResponse(audio_buffer, media_type="audio/mp3")

@app.post('/choose-meditation/')
async def choose_meditation(name, age, gender, struggle, emotion, duration, theme):
    user_prompt = prompt.format(
        user_name = name,
        query = theme,
        user_age = age,
        user_gender = gender,
        user_struggle = struggle,
        user_emotion = emotion,
        meditation_length = duration
    )

    meditation_script = chain.invoke(user_prompt)
    return {'meditation_script': meditation_script["result"]}

@app.post('/create-image')
async def create_image(script: str):
    prompt = f'''
    Analyze the following meditation script to write a one-sentence DALL-E prompt to generate a 16:9 cover image.
    {script}
    '''
    gpt_model = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
    gpt_input = Inputs.get_text_input(input_id="", raw_text=prompt)
    gpt_output = inference(gpt_model, gpt_input)
    
    dalle_model = 'https://clarifai.com/openai/dall-e/models/dall-e-3'
    dalle_input = Inputs.get_text_input(input_id="", raw_text=gpt_output.text.raw)
    dalle_output = inference(dalle_model, dalle_input)

    image_buffer = io.BytesIO(dalle_output.image.base64)
    return StreamingResponse(image_buffer, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
