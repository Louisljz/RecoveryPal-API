import requests

text = 'hi i love roses'
url = f'http://127.0.0.1:8000/text-to-speech/?text={text}'
response = requests.post(url)
if response.status_code == 200:
    with open("voice.mp3", "wb") as file:
        file.write(response.content)
    print('audio file saved!')