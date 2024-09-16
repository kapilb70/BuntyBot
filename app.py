import os
import gradio as gr
from pydub import AudioSegment
import speech_recognition as sr
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.Conversation import Conversation

# Set the API key directly (for testing purposes; not recommended for production)
API_KEY = "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"

# Initialize the GroqModel with the provided API key
llm = GroqModel(api_key=API_KEY)

# Create a SimpleConversationAgent with the GroqModel
agent = SimpleConversationAgent(llm=llm, conversation=Conversation())

# Define the function to handle both text and audio input
def converse(history, input_text, audio_file):
    if audio_file is not None:
        # Handle audio input - Pre-process and convert audio file to text
        audio = AudioSegment.from_file(audio_file)
        # Convert to mono channel
        audio = audio.set_channels(1)
        # Normalize volume
        audio = audio.normalize()
        # Export the preprocessed audio to a new file
        audio.export("processed_audio.wav", format="wav")

        # Use the speech recognition library to transcribe the audio
        recognizer = sr.Recognizer()
        with sr.AudioFile("processed_audio.wav") as source:
            audio_data = recognizer.record(source)
            try:
                input_text = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                history.append(("Audio Input", "Sorry, I could not understand the audio."))
                return history
            except sr.RequestError:
                history.append(("Audio Input", "Sorry, there was an issue with the speech recognition service."))
                return history
    elif input_text:
        # If text input is provided, use it directly
        input_text = input_text
    else:
        history.append(("", "Please provide either a text input or an audio file."))
        return history

    # Process the input text through the conversation agent
    result = agent.exec(input_text)
    history.append((input_text, result))
    return history

# Function to reset inputs and conversation history
def reset_all():
    return [], gr.update(value=""), gr.update(value=None)

# Customize the Gradio interface with both text and audio inputs
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; color: #4A90E2;'>🤖 Ask Me Anything!</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 1.2em;'>Type your message or upload an audio file, and I'll do my best to assist you!</p>")

    # Set initial height for the chatbot
    chatbot = gr.Chatbot(label="Conversation", height=200)  # Initial height set to 200 pixels

    with gr.Row():
        with gr.Column(scale=7):
            text_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", show_label=True)
        with gr.Column(scale=3):
            upload_button = gr.Audio(type="filepath", label="Or Upload Audio", interactive=True)

    submit_button = gr.Button("Submit", variant="primary")
    reset_button = gr.Button("Reset")

    # Link the function to the inputs and outputs
    submit_button.click(converse, inputs=[chatbot, text_input, upload_button], outputs=chatbot)
    submit_button.click(reset_all, outputs=[chatbot, text_input, upload_button])  # Reset inputs after submission

    # Enable "Enter" key to trigger the submit action
    text_input.submit(converse, inputs=[chatbot, text_input, upload_button], outputs=chatbot)
    text_input.submit(reset_all, outputs=[chatbot, text_input, upload_button])  # Reset inputs after submission

    reset_button.click(fn=reset_all, outputs=[chatbot, text_input, upload_button])  # Reset everything

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
