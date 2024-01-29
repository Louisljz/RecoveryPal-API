## RecoveryPal .
This app was developed for NextGen GPT AI Hackathon with Clarifai hosted by Lablab.ai

## Table of Contents
- [Introduction](#introduction)
- [API Documentation](#api-documentation)
- [App Features](#appfeatures)
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
## App Features
**Emotional Check-In**

RecoverPal facilitates a daily emotional check-in, allowing users to monitor and track their emotional state over time. T
his feature provides valuable insights into the user's well-being throughout their recovery journey. 
Regular check-ins foster self-awareness, helping individuals identify triggers and patterns associated with their addiction. 
By understanding their emotional landscape, users can proactively manage stressors and maintain emotional balance.

**Customized Affirmations**

The app utilizes OpenAI GPT-4 Turbo to generate personalized affirmations on a daily basis, tailored to each user's unique needs and progress in recovery. 
Affirmations play a crucial role in building and reinforcing positive self-perceptions. For individuals recovering from addiction, these affirmations
serve as empowering reminders of their strengths, fostering a positive mindset and resilience in the face of challenges.

**Journaling**

RecoverPal encourages users to journal their thoughts and emotions. For newcomers to journaling, the app employs GPT-4 Turbo to offer custom prompts, 
guiding users in their self-reflection and emotional processing. Journaling provides an outlet for users to express their thoughts and feelings in a 
safe space. This reflective practice aids in uncovering underlying emotions and triggers, promoting self-discovery and helping individuals make 
informed decisions on their recovery journey.

**Art Integration**

Recognizing the therapeutic power of art, RecoverPal allows users to submit their own pieces or draw on the whiteboard. 
GPT-4 Vision processes and interprets user-submitted art, providing an additional layer of emotional understanding. 
Expressing emotions through art can be particularly impactful for individuals in recovery, offering a non-verbal means of communication. 
The interpretation of art submissions adds an extra dimension to self-reflection, aiding users in gaining insights into their emotional states.

**Customized Meditation**

Users can select a meditation theme and create a personalized meditation based on their emotional state, journal entries, gender, age, and 
specific struggles. The meditation is then delivered through Clarifai Text-to-Speech for a tailored and soothing experience. Meditation 
serves as a valuable tool for individuals in recovery, promoting relaxation, mindfulness, and stress reduction. By customizing meditations
based on individual inputs, RecoverPal provides a targeted approach to address specific challenges, offering users a supportive and 
calming practice to integrate into their daily routines.

## Docker setup
1. `docker build -t gcr.io/recovery-pal/api:latest .`
2. `docker push gcr.io/recovery-pal/api:latest`

Note: rag-evaluation code using tonic-validate is on tonic branch

[Link to Mobile App repository](https://github.com/Louisljz/RecoveryPal-App)
