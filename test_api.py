import requests

def post_feelings(question, answer):
    global conversation
    conversation +=  f'Question: {question}\n Answer: {answer}\n '
    response = requests.post(f'http://127.0.0.1:8000/process-feelings?conversation={conversation}')
    return response.json()

def create_prompts(mood, conversation):
    response = requests.post(f'http://127.0.0.1:8000/generate-prompts?mood={mood}&conversation={conversation}')
    return response.json()

conversation = ''
question = 'How do you feel today?'
answer = input(question + '\n')
response = post_feelings(question, answer)
for i in range(3):
    question = response['follow_up_question']
    if question:
        answer = input(question + '\n')
        response = post_feelings(question, answer)
    else:
        break

mood = response['mood_scale']
if mood:
    print('Mood Scale: ', mood)
    prompts = create_prompts(mood, conversation)
    print(prompts)

else:
    print('Unable to identify mood!')
