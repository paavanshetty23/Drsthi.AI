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

def main(input_image_path):
    """Main function to process the image and generate a refined UI analysis prompt."""
    try:
        cnn_model = create_cnn_model((512, 512, 3))
        
        processed_image, original_image = process_image(input_image_path, cnn_model)
        dominant_colors = extract_color_palette(processed_image)
        extracted_text, text_positions = extract_text_and_positions(input_image_path)

        print("Generating structured prompt...")
        initial_prompt = generate_prompt(dominant_colors, extracted_text, text_positions)

        print("Refining prompt with Groq API...")
        refined_prompt = refine_prompt_with_groq(initial_prompt)

        print("Comparing images...")
        similarity = compare_images(original_image, processed_image)
        print(f"Image similarity: {similarity:.2f}")

        print("Encoding images...")
        original_encoded = encode_image(original_image)
        processed_encoded = encode_image(processed_image)

        result = {
            "refined_prompt": refined_prompt,
            "dominant_colors": dominant_colors.tolist(),
            "extracted_text": extracted_text,
            "text_positions": text_positions,
            "original_image": original_encoded,
            "processed_image": processed_encoded,
            "image_similarity": float(similarity)
        }

        print("Saving result to JSON file...")
        with open("ui_analysis_result.json", "w") as f:
            json.dump(result, f)

        print("Analysis complete. Results saved to ui_analysis_result.json")
        return result
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    input_image_path = r"C:\Users\paava\Desktop\Academics\Projects\Distri.ai\example_2.png"
    main(input_image_path)
