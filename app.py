import streamlit as st
import base64
from PIL import Image
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import json

# Import your existing functions here
from main import (create_cnn_model, process_image, extract_color_palette, 
                  extract_text_and_positions, generate_prompt, refine_prompt_with_groq,
                  compare_images, encode_image, flatten_json, generate_structured_prompt,
                  further_refine_prompt_with_llm)

# Load environment variables
load_dotenv()

# Set the page title and favicon
st.set_page_config(page_title="Drsthi.AI", page_icon="👁️")

# Function to load and encode the background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background image
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to create a download link for text files
def get_download_link(text, filename, link_text):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

# Main Streamlit app
def main():
    st.title("Drsthi.AI")
    st.markdown("""
    <div class="description-container">
        Drsthi.AI is a tool designed to convert website UI images into LLM-convenient prompts. 
        Upload an image of your website's UI, and Drsthi.AI will analyze it and generate a detailed prompt 
        that can be used with large language models (LLMs) for further processing and analysis.
    </div>
    """, unsafe_allow_html=True)

    # Custom CSS for the chat-like interface, description container, and copy button
    st.markdown("""
    <style>
    .chat-container {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .description-container {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #075E54;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        text-align: center;
        padding: 10px;
    }
    .copy-btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 5px 10px;
        cursor: pointer;
        border-radius: 5px;
        font-size: 12px;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # File uploader for image or data
    uploaded_file = st.file_uploader("Upload an image or data file", type=['png', 'jpg', 'jpeg', 'txt'])

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            # Process the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Save the uploaded image temporarily
            temp_image_path = "temp_image.png"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the image
            cnn_model = create_cnn_model((512, 512, 3))
            processed_image, original_image = process_image(temp_image_path, cnn_model)
            dominant_colors = extract_color_palette(processed_image)
            extracted_text, text_positions = extract_text_and_positions(temp_image_path)
            
            # Generate and refine prompt
            initial_prompt = generate_prompt(dominant_colors, extracted_text, text_positions)
            refined_prompt = refine_prompt_with_groq(initial_prompt)
            
            # Display refined prompt and copy button
            st.markdown(f'<div class="bot-message" id="refined-prompt">{refined_prompt}</div>', unsafe_allow_html=True)
            st.markdown('<button class="copy-btn" onclick="copyRefinedPrompt()">Copy</button>', unsafe_allow_html=True)
            
            # JavaScript for copying text to clipboard
            st.markdown("""
            <script>
            function copyRefinedPrompt() {
                const promptText = document.getElementById("refined-prompt").innerText;
                navigator.clipboard.writeText(promptText).then(function() {
                    alert('Text copied to clipboard');
                }, function(err) {
                    alert('Failed to copy text: ', err);
                });
            }
            </script>
            """, unsafe_allow_html=True)
            
            # Generate text file with image info
            image_info = f"Extracted Text: {extracted_text}\n\nText Positions: {json.dumps(text_positions, indent=2)}\n\nDominant Colors: {dominant_colors.tolist()}"
            
            # Provide download link for the text file
            st.markdown(get_download_link(image_info, "image_analysis.txt", "Download Image Analysis"), unsafe_allow_html=True)
            
            # Clean up temporary file
            os.remove(temp_image_path)
            
        else:
            # For text files, read and display content
            string_data = uploaded_file.getvalue().decode("utf-8")
            st.text_area("File Content", string_data, height=300)
            
            # Dummy response for text analysis
            dummy_response = "The uploaded text file contains information about user interface elements. It describes various components such as buttons, forms, and navigation menus. The text suggests a focus on user experience and accessibility in the design process."
            
            st.markdown(f'<div class="bot-message" id="text-response">{dummy_response}</div>', unsafe_allow_html=True)
            st.markdown('<button class="copy-btn" onclick="copyTextResponse()">Copy</button>', unsafe_allow_html=True)

            # JavaScript for copying text response to clipboard
            st.markdown("""
            <script>
            function copyTextResponse() {
                const textResponse = document.getElementById("text-response").innerText;
                navigator.clipboard.writeText(textResponse).then(function() {
                    alert('Text copied to clipboard');
                }, function(err) {
                    alert('Failed to copy text: ', err);
                });
            }
            </script>
            """, unsafe_allow_html=True)

    # Footer with clickable message and links
    st.markdown("""
    <div class="footer">
        <p>Want to improve the Product or Collaborate?</p>
        <a href="mailto:paavanshetty2004@gmail.com" style="color: white; text-decoration: none;">paavanshetty2004@gmail.com</a> | 
        <a href="https://github.com/paavanshetty23" target="_blank" style="color: white; text-decoration: none;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# Set background image
set_background('image.jpg')

if __name__ == "__main__":
    main()