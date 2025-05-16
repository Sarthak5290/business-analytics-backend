from flask import Flask, request, jsonify
from flask_cors import CORS
from phi.agent import Agent
from phi.model.google import Gemini
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

canvas_agent = Agent(
    name="Canvas Generator",
    model=Gemini(id="gemini-1.5-flash"),
    instructions=[
        "You are an AI business model canvas generator.",
        "Your task is to create a Business Model Canvas for a startup idea.",
        "For each of the following points, provide exactly 3 bullet points (each bullet must not exceed 5 words):",
        "1. Problem",
        "2. Solution",
        "3. Unique Value Propositions",
        "4. Key Metrics",
        "5. Unfair Advantage",
        "6. Distribution Channels",
        "7. Customer Agents",
        "8. Cost Structure",
        "9. Revenue Streams",
        "10. Technical Overview",
        "Return your answer as a valid JSON object with the keys:",
        "\"Problem\", \"Solution\", \"Unique Value Propositions\", \"Key Metrics\", \"Unfair Advantage\",",
        "\"Distribution Channels\", \"Customer Agents\", \"Cost Structure\", \"Revenue Streams\", \"Technical Overview\"."
    ]
)

def clean_agent_response(raw_text):
    """
    Removes markdown code fences (e.g. ```json ... ```) from the agent response.
    """
    raw_text = raw_text.strip()
    # Check if the text starts with a code fence.
    if raw_text.startswith("```"):
        # Split the response into lines.
        lines = raw_text.splitlines()
        # Remove the first line if it starts with ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove the last line if it ends with ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        return cleaned
    return raw_text

@app.route("/canvas", methods=["POST"])
def generate_canvas():
    try:
        data = request.json
        startup_description = data.get("startup_description", "")
        
        # Build the prompt instructing the agent to generate the Business Model Canvas.
        canvas_prompt = f"""Create a Business Model Canvas for the following startup idea:
{startup_description}

For each of the following points, provide exactly 3 bullet points (each bullet not exceeding 5 words):

1. Problem
2. Solution
3. Unique Value Propositions
4. Key Metrics
5. Unfair Advantage
6. Distribution Channels
7. Customer Agents
8. Cost Structure
9. Revenue Streams
10. Technical Overview

Return your answer as a valid JSON object with the keys:
"Problem", "Solution", "Unique Value Propositions", "Key Metrics", "Unfair Advantage", "Distribution Channels", "Customer Agents", "Cost Structure", "Revenue Streams", "Technical Overview".
"""
        agent_response = canvas_agent.run(canvas_prompt)
        raw_content = agent_response.content
        
        # Clean the response to remove markdown code fences.
        cleaned_content = clean_agent_response(raw_content)
        
        # Attempt to parse the cleaned content as JSON.
        try:
            canvas_data = json.loads(cleaned_content)
        except Exception as parse_error:
            return jsonify({
                "error": "Failed to parse agent response as JSON.",
                "raw": raw_content
            }), 500
        
        return jsonify({
            "response": canvas_data,
            "status": "canvas_generated",
            "input": startup_description
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=4001)
