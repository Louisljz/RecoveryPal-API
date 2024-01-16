from clarifai.client.model import Model
from clarifai.client.input import Inputs
from dotenv import load_dotenv

load_dotenv()

with open('test.jpg', "rb") as f:
    img = f.read()

def predict(model_url, inputs, params={}):
    model_prediction = Model(model_url).predict(inputs=[inputs], inference_params=params)
    return model_prediction.outputs[0].data

model_url = "https://clarifai.com/openai/chat-completion/models/openai-gpt-4-vision"
prompt = "describe the image in one sentence"
inputs = Inputs.get_multimodal_input(input_id="", image_bytes=img, raw_text=prompt)

print('GPT-4-VISION')
print(predict(model_url, inputs).text.raw)

model_url = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo"
prompt = "define photosynthesis in one sentence"
inputs = Inputs.get_text_input(input_id="", raw_text=prompt)

print('GPT-4-TURBO')
print(predict(model_url, inputs).text.raw)

model_url = 'https://clarifai.com/openai/dall-e/models/dall-e-3'
prompt = "A cozy cabin in the woods surrounded by colorful autumn leaves"
inputs = Inputs.get_text_input(input_id="", raw_text=prompt)

print('DALL-E')
image = predict(model_url, inputs).image.base64
with open('drawing.jpg', 'wb') as f:
    f.write(image)
print('Drawing saved!')

model_url = 'https://clarifai.com/openai/tts/models/openai-tts-1'
prompt = "Hello I am GPT Bot"
inputs = Inputs.get_text_input(input_id="", raw_text=prompt)

print('OpenAI TTS')
audio = predict(model_url, inputs).audio.base64
with open('voice.mp3', 'wb') as f:
    f.write(audio)
print('Audio saved!')