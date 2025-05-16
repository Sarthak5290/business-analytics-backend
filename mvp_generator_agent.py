import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from flask import Flask, request, jsonify
from flask_cors import CORS
from phi.agent import Agent
from phi.model.google import Gemini
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

# Agent to generate a detailed HTML prompt with UI/UX design guidelines.
html_prompt_agent = Agent(
    name="HTML Prompt Generator",
    model=Gemini(id="gemini-1.5-flash"),
    instructions=[
        "You are an AI prompt generator for HTML code creation with an emphasis on UI/UX design.",
        "Your task is to generate a detailed prompt that describes the desired UI/UX features for a website.",
        "The prompt should include a comprehensive description of layout, navigation, color scheme, typography, responsiveness, interactivity, animations, and form design.",
        "It should also specify modern design principles and user-friendly features that enhance the overall experience.",
        "Return the prompt as clear, structured text with detailed instructions for generating high-quality HTML code.",
    ],
)

# Agent to generate the actual HTML code based on a detailed prompt.
html_code_agent = Agent(
    name="HTML Code Generator",
    model=Gemini(id="gemini-1.5-flash"),
    instructions=[
        "You are an AI HTML code generator specialized in creating fully functional, modern HTML pages with excellent UI/UX design.",
        "Your task is to generate an HTML file for a startup validation and business model canvas website.",
        "The HTML code must include inline CSS styling so that all styling is contained within the HTML file.",
        "The HTML should include semantic elements, a navigation bar, hero section, content areas, interactive forms, and a footer.",
        "Incorporate responsive design, modern typography, and clear color schemes as per modern UI/UX best practices.",
        "Output only the HTML code without any additional explanations or markdown formatting.",
    ],
)


def clean_agent_response(raw_text):
    """
    Removes markdown code fences (e.g. ```html ... ```) from the agent response.
    """
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        return cleaned
    return raw_text


@app.route("/generate_html_prompt", methods=["POST"])
def generate_html_prompt():
    try:
        data = request.json
        # Optional: accept an application description from the frontend.
        application_description = data.get(
            "application_description",
            "Startup validation and business model canvas website",
        )

        # Build the prompt input for detailed UI/UX design.
        prompt_input = f"""Generate a detailed HTML generation prompt for an application described as:
{application_description}

Include instructions for a modern UI/UX design with the following details:
- Layout: clear structure with header, hero, content sections, and footer.
- Navigation: intuitive menus and responsive navigation bar.
- Color scheme: modern, appealing colors with good contrast.
- Typography: readable fonts and sizes.
- Responsiveness: mobile-friendly design.
- Interactive elements: forms, buttons, animations.
- Additional features: accessibility considerations and smooth user interactions.

Provide the prompt as clear, structured text.
"""
        agent_response = html_prompt_agent.run(prompt_input)
        return jsonify(
            {"response": agent_response.content, "status": "html_prompt_generated"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate_html_code", methods=["POST"])
def generate_html_code():
    try:
        data = request.json
        # Optional: accept an application description that might further customize the HTML.
        application_description = data.get(
            "application_description",
            "Startup validation and business model canvas website",
        )

        # Build a prompt for generating the actual HTML code.
        code_prompt = f"""Generate a complete HTML file for an application described as:
{application_description}

The HTML should be structured with the following elements:
- A header with a navigation bar.
- A hero section with a clear call-to-action.
- Content sections explaining the features, including forms for idea submission and validation.
- A footer with contact information and links.
- Use semantic HTML tags and ensure the page is responsive.
- Include inline CSS styling so that all style definitions are within the HTML file.
- Incorporate modern UI/UX design principles with clear typography, appealing color schemes, and interactive elements.

Output only the HTML code without any additional text or markdown formatting.
"""
        agent_response = html_code_agent.run(code_prompt)
        raw_html = agent_response.content
        # Clean potential markdown code fences.
        cleaned_html = clean_agent_response(raw_html)
        return jsonify({"response": cleaned_html, "status": "html_code_generated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=4003)
