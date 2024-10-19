import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def analyze_json(json_file):
    """Load and analyze the JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extracting useful data from JSON
    dominant_colors = data.get("dominant_colors", [])
    extracted_text = data.get("extracted_text", "")
    text_positions = data.get("text_positions", [])

    return dominant_colors, extracted_text, text_positions

def generate_structured_prompt(dominant_colors, extracted_text, text_positions):
    """Generate a structured prompt based on the JSON data."""
    color_palette = ", ".join([f"RGB({color[0]}, {color[1]}, {color[2]})" for color in dominant_colors])
    
    # Generate a detailed prompt using the extracted information
    prompt = f"""
    Analyze the following UI elements and provide a detailed description:

    1. **Color Scheme:**
    - Dominant colors: {color_palette}
    - Describe how these colors are used in the UI.

    2. **Layout and Structure:**
    - Describe the overall layout and main sections of the UI.

    3. **Typography and Text Content:**
    - Analyze the text content and its positioning: {json.dumps(text_positions, indent=2)}

    4. **UI Components and Navigation:**
    - Identify and describe key UI elements and the navigation system.

    5. **Content Organization:**
    - Summarize how content is organized and presented.

    6. **Responsive Design and Accessibility:**
    - Speculate on responsive design and comment on accessibility.

    7. **Overall UX:**
    - Provide insights on the user experience and potential improvements.

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

def main(json_file):
    """Main function to load JSON, generate a structured prompt, and refine it with Groq API."""
    try:
        dominant_colors, extracted_text, text_positions = analyze_json(json_file)

        print("Generating structured prompt...")
        initial_prompt = generate_structured_prompt(dominant_colors, extracted_text, text_positions)

        print("Refining prompt with Groq API...")
        refined_prompt = refine_prompt_with_groq(initial_prompt)

        print("Refined Prompt:")
        print(refined_prompt)

        # Save the refined prompt to a file
        with open("refined_prompt.txt", "w") as f:
            f.write(refined_prompt)

        print("Refined prompt saved to refined_prompt.txt")
        return refined_prompt

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    json_file = r"C:\Users\paava\Desktop\Academics\Projects\Distri.ai\ui_analysis_result.json"  # Path to the JSON file generated in the previous step
    main(json_file)
