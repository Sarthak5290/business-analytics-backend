import os
import re
import logging
from langchain_groq import ChatGroq
from flask import (
    Flask,
    render_template_string,
    request,
    send_file,
    redirect,
    url_for,
    jsonify,
)
from flask_cors import CORS
import subprocess
import imgkit
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Get Groq API Key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.warning(
        "GROQ_API_KEY not found in environment variables. Please set it to use the Groq API."
    )

# Initialize Langchain ChatGroq with a supported model
llm = None
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_retries=2,
        groq_api_key=GROQ_API_KEY,
    )
except Exception as e:
    logging.error(f"Error initializing Groq LLM: {e}")


def generate_mermaid_code(project_description):
    """Generates Mermaid code from a project description using Groq LLM."""
    if not llm:
        logging.error("Groq LLM not initialized. Cannot generate Mermaid code.")
        return "Error: Groq API not properly configured. Please check your API key and model settings."

    try:
        # Improved prompt with explicit instructions for Mermaid code block formatting
        prompt = f"""
        Create a Mermaid diagram code that visually represents the following project: 
        
        {project_description}
        
        Focus on the key components and their relationships. The diagram should illustrate the architecture and dependencies.
        
        IMPORTANT: Your response MUST include a Mermaid code block formatted exactly like this:
        ```mermaid
        [your mermaid code here]
        ```
        
        The code should be valid Mermaid syntax. Use graph TD or flowchart TD for top-down diagrams.
        """

        logging.info(f"Sending prompt to Groq: {prompt[:100]}...")
        response = llm.invoke(prompt)
        logging.info(f"Received response from Groq: {response.content[:100]}...")
        return response.content
    except Exception as e:
        logging.error(f"Error generating Mermaid code: {e}")
        return f"Error generating Mermaid code: {str(e)}"


def extract_mermaid_code(text):
    """Extracts Mermaid code from a text string using regex."""
    if not text:
        logging.warning("Empty text provided to extract_mermaid_code")
        return None

    # Log the input to help debug extraction issues
    logging.info(
        f"Extracting Mermaid code from text (first 100 chars): {text[:100]}..."
    )

    # Try multiple regex patterns to be more robust
    patterns = [
        r"```mermaid\s*(.*?)\s*```",  # Standard markdown mermaid block
        r"`{3}mermaid\s*(.*?)\s*`{3}",  # Alternative syntax
        r"<div\s+class=\"mermaid\">\s*(.*?)\s*<\/div>",  # HTML div with mermaid class
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            mermaid_code = match.group(1).strip()
            logging.info(
                f"Successfully extracted Mermaid code (first 50 chars): {mermaid_code[:50]}..."
            )
            return mermaid_code

    # If we get here, we didn't find any Mermaid code
    logging.warning("No Mermaid code found in the text")

    # As a fallback, if the text looks like Mermaid code without the markers, return it
    if "graph " in text or "flowchart " in text:
        logging.info(
            "No specific Mermaid block found, but text appears to be Mermaid code. Using as is."
        )
        return text

    return None


def save_to_readme(mermaid_code):
    """Saves the Mermaid code to a readme.md file."""
    try:
        with open("readme.md", "w") as f:
            f.write("## Mermaid Diagram\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_code)
            f.write("\n```\n")
        logging.info("Mermaid code saved to readme.md")
    except Exception as e:
        logging.error(f"Error saving to readme.md: {e}")
        return False
    return True


def detect_wkhtmltoimage_path():
    """Detect the path to wkhtmltoimage based on OS."""
    # Try common locations
    possible_paths = [
        # Windows
        "C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltoimage.exe",
        # Mac
        "/usr/local/bin/wkhtmltoimage",
        # Linux
        "/usr/bin/wkhtmltoimage",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # If we couldn't find it in common locations, try to find it in PATH
    try:
        result = subprocess.run(
            ["which", "wkhtmltoimage"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None


def convert_readme_to_png(readme_file="readme.md", output_png="mermaid_diagram.png"):
    """Converts the readme.md file containing Mermaid code to a PNG image."""
    try:
        # Try to find wkhtmltoimage
        wkhtmltopdf_path = detect_wkhtmltoimage_path()

        if not wkhtmltopdf_path:
            logging.error(
                "wkhtmltoimage not found. Please install it or specify the path."
            )
            return False

        config = imgkit.config(wkhtmltoimage=wkhtmltopdf_path)

        # Read the mermaid code from readme.md
        with open(readme_file, "r") as f:
            readme_content = f.read()

        # Extract mermaid code
        mermaid_code = extract_mermaid_code(readme_content)

        if not mermaid_code:
            logging.error("No Mermaid code found in readme.md")
            return False

        # Create an HTML template with mermaid code
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Mermaid Diagram</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10.2.0/dist/mermaid.min.js"></script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    mermaid.initialize({{ startOnLoad: true }});
                }});
            </script>
        </head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
        </body>
        </html>
        """

        # Save the HTML to a temporary file
        with open("temp.html", "w") as temp_file:
            temp_file.write(html_content)

        imgkit.from_file("temp.html", output_png, config=config)
        os.remove("temp.html")

        logging.info(f"Mermaid diagram saved to {output_png}")
        return True
    except Exception as e:
        logging.error(f"Error converting to PNG: {e}")
        return False


@app.route("/", methods=["GET"])
def index():
    """Main page with form to input project description."""
    return render_template_string(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Mermaid Diagram Generator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            textarea {
                width: 100%;
                height: 150px;
                margin-bottom: 10px;
                padding: 8px;
            }
            button {
                padding: 10px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Mermaid Diagram Generator</h1>
        <p>Enter a project description, and the application will generate a Mermaid diagram for you.</p>
        
        <form action="/mermaid" method="post">
            <textarea name="project_description" placeholder="Describe your project here..."></textarea>
            <button type="submit">Generate Diagram</button>
        </form>
    </body>
    </html>
    """
    )


@app.route("/mermaid", methods=["GET", "POST"])
def mermaid_endpoint():
    """Handle both GET and POST requests for mermaid endpoint."""
    project_description = ""
    mermaid_code = None
    png_available = False
    error_message = None

    # Handle POST request (from form or API)
    if request.method == "POST":
        # Log the request content type and data for debugging
        logging.info(f"Request Content-Type: {request.content_type}")

        # Check if it's a JSON payload (API) or form data
        if request.is_json:
            try:
                data = request.get_json()
                logging.info(f"Received JSON data: {data}")

                if not data or "project_description" not in data:
                    error_response = {
                        "error": "JSON payload with 'project_description' key required."
                    }
                    logging.error(f"Invalid JSON data: {error_response}")
                    return jsonify(error_response), 400

                project_description = data["project_description"]
            except Exception as e:
                logging.error(f"Error parsing JSON: {e}")
                return jsonify({"error": f"Error parsing JSON: {str(e)}"}), 400
        else:
            # Handle form submission
            project_description = request.form.get("project_description", "")
            logging.info(
                f"Received form data, project_description: {project_description[:50]}..."
            )

            if not project_description:
                return redirect(url_for("index"))

        if not GROQ_API_KEY or not llm:
            error_message = "Groq API not properly configured. Please check your API key and model settings."
            logging.error(error_message)
        else:
            # Generate Mermaid code
            logging.info(
                f"Generating Mermaid code for project: {project_description[:50]}..."
            )
            mermaid_text = generate_mermaid_code(project_description)

            if mermaid_text and not mermaid_text.startswith("Error:"):
                mermaid_code = extract_mermaid_code(mermaid_text)

                if mermaid_code:
                    logging.info("Mermaid code extracted successfully")
                    save_to_readme(mermaid_code)
                    if convert_readme_to_png():
                        png_available = True
                    else:
                        logging.warning("Failed to convert readme to PNG")
                        png_available = False
                else:
                    logging.error("Failed to extract Mermaid code from response")
                    mermaid_code = "Mermaid code not found in the response."
                    error_message = (
                        "Failed to extract Mermaid code from the AI response."
                    )
            else:
                error_message = (
                    mermaid_text if mermaid_text else "Failed to generate Mermaid code."
                )
                logging.error(f"Error from Groq API: {error_message}")

    # Handle GET request (showing saved results)
    elif request.method == "GET":
        logging.info("Handling GET request to /mermaid")
        # Check if readme.md exists and try to load it
        if os.path.exists("readme.md"):
            logging.info("Found existing readme.md")
            with open("readme.md", "r") as f:
                readme_content = f.read()
            mermaid_code = extract_mermaid_code(readme_content)
            png_available = os.path.exists("mermaid_diagram.png")
        else:
            # If no saved diagram, redirect to the form
            logging.info("No existing readme.md, redirecting to index")
            return redirect(url_for("index"))

    # Create a response HTML with enhanced debugging and error handling
    html_response = render_template_string(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Mermaid Diagram Generator</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .error {
                    color: red;
                    padding: 10px;
                    background-color: #ffeeee;
                    border-radius: 4px;
                    margin-bottom: 15px;
                }
                .mermaid {
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 4px;
                    margin: 15px 0;
                }
                pre {
                    white-space: pre-wrap;
                }
                a {
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 15px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }
                a:hover {
                    background-color: #45a049;
                }
                img {
                    max-width: 100%;
                    margin-top: 20px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10.2.0/dist/mermaid.min.js"></script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    mermaid.initialize({ 
                        startOnLoad: true,
                        theme: 'default',
                        securityLevel: 'loose',
                        logLevel: 'info',
                        flowchart: { 
                            htmlLabels: true,
                            curve: 'linear'
                        }
                    });
                    
                    // Force re-render after a short delay to ensure the diagram renders
                    setTimeout(function() {
                        try {
                            mermaid.init(undefined, document.querySelectorAll('.mermaid'));
                            console.log("Mermaid re-initialization complete");
                        } catch (e) {
                            console.error("Error re-initializing Mermaid:", e);
                        }
                    }, 1000);
                });
            </script>
        </head>
        <body>                                 
            <h1>Mermaid Diagram Generator</h1>
            
            {% if error_message %}
                <div class="error">
                    <p>{{ error_message }}</p>
                </div>
            {% endif %}

            {% if project_description %}
                <h2>Project Description:</h2>
                <p>{{ project_description }}</p>
            {% endif %}

            {% if mermaid_code %}
                <h2>Generated Mermaid Diagram:</h2>
                <div class="mermaid">
                    {{ mermaid_code }}
                </div>
                
                <!-- Fallback display of the raw code for debugging -->
                <h3>Raw Mermaid Code (for debugging):</h3>
                <pre>{{ mermaid_code }}</pre>
            {% endif %}

            {% if png_available %}
                <h2>Mermaid Diagram (PNG):</h2>
                <img src="/get_png" alt="Mermaid Diagram">
            {% endif %}

            <a href="/">Generate Another Diagram</a>
        </body>
        </html>
    """,
        mermaid_code=mermaid_code,
        png_available=png_available,
        project_description=project_description,
        error_message=error_message,
    )

    # Return the HTML response with the proper Content-Type header
    return html_response, 200, {"Content-Type": "text/html"}


@app.route("/get_png")
def get_png():
    """Serves the generated PNG image."""
    try:
        return send_file("mermaid_diagram.png", mimetype="image/png")
    except FileNotFoundError:
        return "PNG image not found.", 404


@app.route("/api/status")
def api_status():
    """API endpoint to check if the service is running and API key is configured."""
    status = {
        "service": "running",
        "api_key_configured": bool(GROQ_API_KEY),
        "llm_initialized": llm is not None,
    }
    return jsonify(status)


if __name__ == "__main__":
    if not GROQ_API_KEY:
        logging.warning(
            "====================================================================="
        )
        logging.warning(
            "GROQ_API_KEY not set! Please set it in your environment variables."
        )
        logging.warning("You can set it by running: export GROQ_API_KEY=your_api_key")
        logging.warning("Or by creating a .env file with GROQ_API_KEY=your_api_key")
        logging.warning(
            "====================================================================="
        )
    app.run(debug=True)
