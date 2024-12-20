import os
import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import pytesseract
import json
import base64
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from collections import defaultdict

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def create_cnn_model(input_shape):
    """Create a CNN model for image processing."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
    return model

def process_image(image_path, cnn_model):
    """Process the image using the CNN model."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    resized_image = cv2.resize(image, (512, 512))
    normalized_image = resized_image / 255.0
    processed_image = cnn_model.predict(np.expand_dims(normalized_image, axis=0))[0]
    processed_image = (processed_image * 255).astype(np.uint8)
    return processed_image, resized_image

def enhance_image_for_text(image):
    """Enhance image for better text extraction."""
    pil_image = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(2)  # Increase contrast
    return np.array(enhanced_image)

def extract_color_palette(image, num_colors=5):
    """Extract the dominant colors from the image using KMeans clustering."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def extract_text_and_positions(image_path):
    """Extract text and positions from the image using OCR."""
    image = cv2.imread(image_path)
    enhanced_image = enhance_image_for_text(image)  # Enhance for text extraction

    custom_config = r'--oem 3 --psm 6'
    text_data = pytesseract.image_to_data(enhanced_image, config=custom_config, output_type='dict')

    extracted_text = " ".join([word for word in text_data['text'] if word.strip()])
    text_positions = [
        {
            "text": text_data['text'][i],
            "position": (text_data['left'][i], text_data['top'][i]),
            "confidence": text_data['conf'][i]
        }
        for i in range(len(text_data['text']))
        if int(text_data['conf'][i]) > 60 and text_data['text'][i].strip()
    ]
    return extracted_text, text_positions

def generate_prompt(dominant_colors, extracted_text, text_positions):
    """Generate a structured prompt for UI analysis."""
    color_palette = ", ".join([f"RGB({color[0]}, {color[1]}, {color[2]})" for color in dominant_colors])
    
    prompt = f"""Analyze the following UI elements and provide a detailed description:

    1. Color Scheme:
    - Dominant colors: {color_palette}
    - Describe how these colors are used in the UI

    2. Layout and Structure:
    - Describe the overall layout and main sections

    3. Typography and Text Content:
    - Analyze the text content and its positioning:
    {json.dumps(text_positions, indent=2)}

    4. UI Components and Navigation:
    - Identify and describe key UI elements and the navigation system

    5. Content Organization:
    - Summarize how content is organized and presented

    6. Responsive Design and Accessibility:
    - Speculate on responsive design and comment on accessibility

    7. Overall UX:
    - Provide insights on the user experience and potential improvements

    Based on this analysis, generate a comprehensive description of the UI, its design principles, and potential use cases.
    """
    return prompt

def refine_prompt_with_groq(prompt):
    """Refine the prompt using Groq's API."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are an AI assistant specializing in UI/UX analysis and description."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2048
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error refining prompt: {e}")
        return prompt

def compare_images(original_image, processed_image):
    """Compare the original and processed images."""
    original_image = cv2.resize(original_image, (512, 512))
    processed_image = cv2.resize(processed_image, (512, 512))
    diff = cv2.absdiff(original_image.astype(np.float32), processed_image.astype(np.float32))
    return 1 - np.mean(diff) / 255.0

def encode_image(image):
    """Encode image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def flatten_json(nested_json, prefix=''):
    """Flatten nested JSON structure."""
    flattened = {}
    for key, value in nested_json.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_json(value, new_key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    flattened.update(flatten_json(item, f"{new_key}[{i}]"))
                else:
                    flattened[f"{new_key}[{i}]"] = item
        else:
            flattened[new_key] = value
    return flattened

def generate_structured_prompt(flattened_data):
    """Generate a structured prompt from flattened JSON data."""
    sections = defaultdict(list)
    
    for key, value in flattened_data.items():
        if key.startswith("dominant_colors"):
            sections["Color Scheme"].append(f"- {key}: {value}")
        elif key.startswith("text_positions"):
            sections["Typography and Text Content"].append(f"- {key}: {value}")
        elif key.startswith("extracted_text"):
            sections["Content"].append(f"- {value}")
        elif key.startswith("image_similarity"):
            sections["Image Analysis"].append(f"- Similarity: {value:.2f}")
    
    prompt = "Analyze the following UI elements and provide a detailed description:\n\n"
    
    for section, items in sections.items():
        prompt += f"{section}:\n"
        prompt += "\n".join(items)
        prompt += "\n\n"
    
    prompt += "Based on this analysis, generate a comprehensive description of the UI, its design principles, and potential use cases."
    
    return prompt

def further_refine_prompt_with_llm(structured_prompt, flattened_result):
    """Send the structured prompt to the LLM for further refinement with more descriptive detail."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"  # Or any other LLM API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        # Ask the LLM to focus on providing detailed descriptions of the color scheme, word placement, etc.
        detailed_prompt = f"""Refine the following UI analysis into a more descriptive, detailed prompt:
        
        {structured_prompt}

        Focus on:
        - Descriptions of the dominant colors and their usage in the UI.
        - Detailed word placement, font analysis, and overall text layout.
        - Visual hierarchy and spacing between elements.
        - UX suggestions for improvement.

        Use the following extracted data for additional context:
        {json.dumps(flattened_result, indent=2)}

        Provide a structured and user-friendly prompt based on the above data.
        """
        
        payload = {
            "model": "llama3-8b-8192",  # Replace with the actual LLM model you are using
            "messages": [
                {"role": "system", "content": "You are an AI assistant specializing in UI/UX analysis and detailed descriptions."},
                {"role": "user", "content": detailed_prompt}
            ],
            "max_tokens": 2048
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error further refining prompt: {e}")
        return structured_prompt

# Usage Example
if __name__ == "__main__":
    # Load a sample image
    image_path = r"C:\Users\paava\Desktop\Academics\Projects\Distri.ai\example_2.png"
    cnn_model = create_cnn_model((512, 512, 3))
    
    processed_image, original_image = process_image(image_path, cnn_model)
    dominant_colors = extract_color_palette(processed_image)
    extracted_text, text_positions = extract_text_and_positions(image_path)
    
    # Generate initial prompt
    prompt = generate_prompt(dominant_colors, extracted_text, text_positions)
    
    # Refine prompt using Groq API
    refined_prompt = refine_prompt_with_groq(prompt)
    
    # Compare original and processed image
    similarity_score = compare_images(original_image, processed_image)
    
    # Encode image for further analysis
    encoded_image = encode_image(processed_image)
    
    # Flatten data for structured prompt
    result_data = {
        "dominant_colors": dominant_colors.tolist(),
        "text_positions": text_positions,
        "extracted_text": extracted_text,
        "image_similarity": similarity_score
    }
    flattened_result = flatten_json(result_data)
    
    # Generate a structured prompt
    structured_prompt = generate_structured_prompt(flattened_result)
    
    # Further refine the prompt with the LLM
    final_refined_prompt = further_refine_prompt_with_llm(structured_prompt, flattened_result)
    
    print("Final Refined Prompt:")
    print(final_refined_prompt)
