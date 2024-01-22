## **RecoveryPal **
This app was developed for NextGen GPT AI Hackathon with Clarifai hosted by Lablab.ai

## Table of Contents
- [Introduction](#introduction)
- [API Documentation](#api-documentation)
  - [Dockerfile](#dockerfile)
  - [app.py](#apppy)
    clarifai_models.py (https://github.com/Louisljz/RecoveryPal-API/blob/main/clarifai_models.py)
  - [meditation_prompt.txt](#meditationprompttxt)
  - [rag-pipeline.ipynb](#ragpipelineipynb)
  - [test_api.py](#testapipy)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Introduction
RecoverPal is an innovative mobile application designed to provide crucial support for individuals on the journey of addiction recovery. 
The app aims to make this challenging process more manageable by offering personalized features and a supportive community. 
With RecoverPal, users can have a constant companion in their pockets to help them navigate the ups and downs of recovery.
## API Documentation

The app.py file contains the main FastAPI application with various endpoints. It utilizes Clarifai models, OpenAI embeddings, and Pinecone vector stores for different functionalities.
/process-feelings: Analyzes user feelings and provides a mood scale and follow-up question.
/generate-prompts: Generates art, journaling, and meditation suggestions based on user input.
/describe-art: Describes the feelings inferred from a given drawing.
/text-to-speech: Converts text-to-speech using an OpenAI TTS model.
/choose-meditation: Recommends a meditation script based on user input.
/create-image: Generates a DALL-E image based on a meditation script.

meditation_prompt.txt
The meditation_prompt.txt file provides a template for generating meditation scripts based on user input. It includes placeholders for user information, struggles, mood, and journal entries.

clarifai_models.py
The clarifai_models.py file includes code for using Clarifai models for image description, text definition, DALL-E image generation, and OpenAI TTS for text-to-speech.

rag-pipeline.ipynb
The rag-pipeline.ipynb Jupyter notebook demonstrates the use of LangChain, Clarifai, OpenAIEmbeddings, and Pinecone to create a retrieval-based question-answering pipeline. It includes an example template for generating a meditation script.

## Docker setup
1. `docker build -t gcr.io/recovery-pal/api:latest .`
2. `docker push gcr.io/recovery-pal/api:latest`

Note: rag-evaluation code using tonic-validate is on tonic branch

[Link to Mobile App repository](https://github.com/Louisljz/RecoveryPal-App)
