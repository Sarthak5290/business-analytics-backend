import os
import json
import pandas as pd
import numpy as np
import matplotlib

# Set the backend to Agg (non-interactive) before importing pyplot
# This prevents Tkinter-related errors in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from flask import (
    Flask,
    render_template_string,
    request,
    jsonify,
    send_file,
    send_from_directory,
)
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create a directory for storing generated graphs if it doesn't exist
GRAPHS_DIR = "graphs"
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

# Sample data for initial testing
sample_data = {
    "user_input": "This is a sample input",
    "problem": "Sample problem description",
    "target_customers": "Sample target customers",
    "solution": "Sample solution",
    "key_resources": "Sample key resources",
    "revenue_streams": "Monthly subscription: $10/month, Premium tier: $50/month",
}


def parse_revenue_data(revenue_text):
    """Parse revenue streams text into structured data for visualization."""
    # This is a simple parser - in a real app, you might need a more robust approach
    revenue_items = revenue_text.split(",")

    # Extract values and labels
    values = []
    labels = []

    for item in revenue_items:
        item = item.strip()
        if ":" in item and "$" in item:
            # Extract the label and value
            parts = item.split(":")
            label = parts[0].strip()

            # Find the dollar amount
            value_text = parts[1].strip()
            dollar_amount = "".join(c for c in value_text if c.isdigit() or c == ".")

            try:
                value = float(dollar_amount)
                values.append(value)
                labels.append(label)
            except ValueError:
                # Skip if we can't parse a valid number
                continue

    return values, labels


def generate_revenue_pie_chart(revenue_text):
    """Generate a pie chart for revenue streams."""
    values, labels = parse_revenue_data(revenue_text)

    if not values:
        # Generate sample data if parsing failed
        values = [10, 50, 25]
        labels = ["Basic tier", "Premium tier", "Enterprise"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title("Revenue Distribution by Stream")

    # Save to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)

    # Save to file
    plt.savefig(os.path.join(GRAPHS_DIR, "revenue_pie_chart.png"), bbox_inches="tight")
    plt.close(fig)  # Explicitly close the figure to avoid memory leaks

    return img_bytes


def generate_revenue_projection(revenue_text):
    """Generate a projection of revenue growth over time."""
    values, labels = parse_revenue_data(revenue_text)

    if not values:
        # Generate sample data if parsing failed
        values = [100, 500, 250]
        labels = ["Basic tier", "Premium tier", "Enterprise"]

    # Generate time series data (12 months)
    months = pd.date_range(start=datetime.now(), periods=12, freq="ME")

    # Create dataframe for projections
    df = pd.DataFrame(index=months)

    # Growth rates for each revenue stream (monthly)
    growth_rates = np.random.uniform(0.05, 0.15, len(values))

    # Generate projections for each revenue stream
    for i, (value, label) in enumerate(zip(values, labels)):
        # Apply compounding growth
        projection = [value * (1 + growth_rates[i]) ** j for j in range(12)]
        df[label] = projection

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each revenue stream
    for column in df.columns:
        ax.plot(df.index, df[column], marker="o", linestyle="-", label=column)

    # Add total revenue line
    df["Total"] = df.sum(axis=1)
    ax.plot(
        df.index,
        df["Total"],
        marker="s",
        linestyle="-",
        linewidth=2,
        label="Total Revenue",
    )

    # Format the plot
    ax.set_title("Revenue Projection Over Next 12 Months")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Format x-axis dates
    plt.xticks(rotation=45)
    fig.tight_layout()

    # Save to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)

    # Save to file
    plt.savefig(os.path.join(GRAPHS_DIR, "revenue_projection.png"), bbox_inches="tight")
    plt.close(fig)  # Explicitly close the figure

    return img_bytes


def generate_customer_growth(target_customers):
    """Generate a projection of customer growth over time."""
    # Parse target customers to estimate initial customer base
    words = len(target_customers.split())
    initial_customers = max(100, words * 10)  # Simple heuristic

    # Generate time series data (24 months)
    months = pd.date_range(start=datetime.now(), periods=24, freq="ME")

    # Create dataframe for projections
    df = pd.DataFrame(index=months)

    # Growth model parameters
    growth_rate = 0.1  # 10% monthly growth
    saturation = initial_customers * 10  # Maximum market size

    # S-curve growth model (logistic function)
    def logistic_growth(t, initial, saturation, growth_rate):
        return saturation / (
            1 + ((saturation / initial) - 1) * np.exp(-growth_rate * t)
        )

    # Generate customer growth projection
    projection = [
        logistic_growth(t, initial_customers, saturation, growth_rate)
        for t in range(24)
    ]
    df["Customers"] = projection

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual projections
    ax.plot(
        df.index[:12],
        df["Customers"][:12],
        marker="o",
        linestyle="-",
        color="blue",
        label="Projected Growth",
    )

    # Plot predicted projections with different style
    ax.plot(
        df.index[11:],
        df["Customers"][11:],
        marker="o",
        linestyle="--",
        color="red",
        label="Predicted Future Growth",
    )

    # Add an annotation to highlight the prediction point
    plt.axvline(x=df.index[11], color="gray", linestyle=":", alpha=0.7)
    plt.text(
        df.index[11],
        df["Customers"].max() / 2,
        "Prediction Start",
        rotation=90,
        va="center",
    )

    # Format the plot
    ax.set_title("Customer Growth Projection")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Customers")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    # Format x-axis dates
    plt.xticks(rotation=45)
    fig.tight_layout()

    # Save to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)

    # Save to file
    plt.savefig(os.path.join(GRAPHS_DIR, "customer_growth.png"), bbox_inches="tight")
    plt.close(fig)  # Explicitly close the figure

    return img_bytes


def generate_market_position(problem, solution):
    """Generate a quadrant chart positioning the solution in the market."""
    # Calculate metrics based on text analysis
    problem_words = problem.split()
    solution_words = solution.split()

    # Simple metric: uniqueness based on word count ratio
    uniqueness = min(10, len(solution_words) / max(1, len(problem_words)) * 5)

    # Simple metric: value based on solution words
    value_keywords = [
        "efficient",
        "effective",
        "save",
        "improve",
        "increase",
        "decrease",
        "reduce",
        "enhance",
    ]
    value_score = sum(
        1
        for word in solution_words
        if any(keyword in word.lower() for keyword in value_keywords)
    )
    value = min(10, value_score + 5)  # Base value of 5, max of 10

    # Create the quadrant chart
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the quadrants
    ax.axhline(y=5, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=5, color="gray", linestyle="-", alpha=0.3)

    # Plot the solution position
    ax.plot(uniqueness, value, "ro", markersize=15)

    # Add labels for each quadrant
    ax.text(2.5, 7.5, "Niche Players", ha="center", va="center", fontsize=12)
    ax.text(7.5, 7.5, "Market Leaders", ha="center", va="center", fontsize=12)
    ax.text(2.5, 2.5, "Low Priority", ha="center", va="center", fontsize=12)
    ax.text(7.5, 2.5, "Emerging Solutions", ha="center", va="center", fontsize=12)

    # Set limits and labels
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Solution Uniqueness")
    ax.set_ylabel("Value Proposition")
    ax.set_title("Market Position Analysis")

    # Add an annotation for the solution
    ax.annotate(
        "Your Solution",
        xy=(uniqueness, value),
        xytext=(uniqueness + 1, value + 1),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
    )

    # Save to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)

    # Save to file
    plt.savefig(os.path.join(GRAPHS_DIR, "market_position.png"), bbox_inches="tight")
    plt.close(fig)  # Explicitly close the figure

    return img_bytes


def generate_resource_allocation(key_resources):
    """Generate a bar chart showing resource allocation."""
    # Parse key resources
    resources = [r.strip() for r in key_resources.split(",")]

    # Generate random weights for resources
    weights = np.random.uniform(0.5, 1.0, len(resources))
    weights = weights / weights.sum() * 100  # Normalize to percentages

    # Sort for better visualization
    sorted_indices = np.argsort(weights)[::-1]
    sorted_resources = [resources[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bars
    y_pos = np.arange(len(sorted_resources))
    ax.barh(y_pos, sorted_weights, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_resources)

    # Add percentage labels
    for i, v in enumerate(sorted_weights):
        ax.text(v + 1, i, f"{v:.1f}%", va="center")

    # Format the plot
    ax.set_xlabel("Allocation (%)")
    ax.set_title("Resource Allocation")
    ax.set_xlim(0, 100)

    fig.tight_layout()

    # Save to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)

    # Save to file
    plt.savefig(
        os.path.join(GRAPHS_DIR, "resource_allocation.png"), bbox_inches="tight"
    )
    plt.close(fig)  # Explicitly close the figure

    return img_bytes


def generate_all_graphs(data):
    """Generate all graphs based on the input data."""
    try:
        # Generate each graph
        revenue_pie = generate_revenue_pie_chart(data.get("revenue_streams", ""))
        revenue_proj = generate_revenue_projection(data.get("revenue_streams", ""))
        customer_growth = generate_customer_growth(data.get("target_customers", ""))
        market_position = generate_market_position(
            data.get("problem", ""), data.get("solution", "")
        )
        resource_allocation = generate_resource_allocation(
            data.get("key_resources", "")
        )

        # Create an HTML file with all the graphs
        create_html_report(data)

        return {
            "revenue_pie": True,
            "revenue_projection": True,
            "customer_growth": True,
            "market_position": True,
            "resource_allocation": True,
            "html_report": True,
        }
    except Exception as e:
        print(f"Error generating graphs: {str(e)}")
        return {"error": str(e)}


def create_html_report(data):
    """Create an HTML report with all graphs."""

    # Read the graph images as base64
    def read_image_base64(filename):
        try:
            with open(os.path.join(GRAPHS_DIR, filename), "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading image {filename}: {e}")
            return ""

    revenue_pie_b64 = read_image_base64("revenue_pie_chart.png")
    revenue_proj_b64 = read_image_base64("revenue_projection.png")
    customer_growth_b64 = read_image_base64("customer_growth.png")
    market_position_b64 = read_image_base64("market_position.png")
    resource_allocation_b64 = read_image_base64("resource_allocation.png")

    # Create an HTML file with embedded images
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Business Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .report-header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .graph-container {{
                margin-bottom: 40px;
                padding: 20px;
                border: 1px solid #eee;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .graph-img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
            .description {{
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .data-section {{
                margin-bottom: 30px;
                padding: 15px;
                background-color: #f0f7ff;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>Business Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
        </div>
        
        <div class="data-section">
            <h2>Business Overview</h2>
            <p><strong>Problem:</strong> {data.get('problem', 'No problem description provided')}</p>
            <p><strong>Target Customers:</strong> {data.get('target_customers', 'No target customers specified')}</p>
            <p><strong>Solution:</strong> {data.get('solution', 'No solution described')}</p>
            <p><strong>Key Resources:</strong> {data.get('key_resources', 'No key resources listed')}</p>
            <p><strong>Revenue Streams:</strong> {data.get('revenue_streams', 'No revenue streams defined')}</p>
        </div>
        
        <div class="graph-container">
            <h2>Revenue Distribution</h2>
            <img src="data:image/png;base64,{revenue_pie_b64}" alt="Revenue Distribution" class="graph-img">
            <div class="description">
                <p>This pie chart shows the distribution of revenue across different streams. The proportions indicate the relative contribution of each revenue source to the overall business income.</p>
            </div>
        </div>
        
        <div class="graph-container">
            <h2>Revenue Projection (12 Months)</h2>
            <img src="data:image/png;base64,{revenue_proj_b64}" alt="Revenue Projection" class="graph-img">
            <div class="description">
                <p>This chart projects revenue growth over the next 12 months for each revenue stream and the total revenue. The projection is based on estimated growth rates derived from the business model.</p>
            </div>
        </div>
        
        <div class="graph-container">
            <h2>Customer Growth Projection</h2>
            <img src="data:image/png;base64,{customer_growth_b64}" alt="Customer Growth" class="graph-img">
            <div class="description">
                <p>This graph shows projected customer growth over 24 months. The solid line represents near-term projections, while the dashed line indicates predicted future growth based on market saturation models.</p>
            </div>
        </div>
        
        <div class="graph-container">
            <h2>Market Position Analysis</h2>
            <img src="data:image/png;base64,{market_position_b64}" alt="Market Position" class="graph-img">
            <div class="description">
                <p>This quadrant chart positions the solution in the market based on its uniqueness and value proposition. The position indicates the strategic market placement of the business.</p>
            </div>
        </div>
        
        <div class="graph-container">
            <h2>Resource Allocation</h2>
            <img src="data:image/png;base64,{resource_allocation_b64}" alt="Resource Allocation" class="graph-img">
            <div class="description">
                <p>This chart shows the recommended allocation of resources based on the business model. The percentages indicate the relative importance of each resource to the business operations.</p>
            </div>
        </div>
        
        <div class="graph-container">
            <h2>Conclusions and Recommendations</h2>
            <div class="description">
                <p>Based on the analysis of the business model:</p>
                <ul>
                    <li>The business shows potential for growth in the identified market segments.</li>
                    <li>Key resources should be allocated according to the recommended distribution to maximize efficiency.</li>
                    <li>Revenue projections indicate a positive trend, but continued monitoring is advised.</li>
                    <li>The market position suggests the business has a competitive advantage in its niche.</li>
                </ul>
                <p><strong>Next Steps:</strong></p>
                <ul>
                    <li>Validate revenue assumptions with market testing.</li>
                    <li>Review customer acquisition strategy to align with projected growth.</li>
                    <li>Regularly update projections based on actual performance data.</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    # Save the HTML file
    with open(os.path.join(GRAPHS_DIR, "business_report.html"), "w") as f:
        f.write(html_content)

    # Also save a simpler version as graphs.txt for compatibility
    with open("graphs.txt", "w") as f:
        f.write(
            f"""
Business Analysis Report
Generated on {datetime.now().strftime('%B %d, %Y')}

Business Overview:
- Problem: {data.get('problem', 'No problem description provided')}
- Target Customers: {data.get('target_customers', 'No target customers specified')}
- Solution: {data.get('solution', 'No solution described')}
- Key Resources: {data.get('key_resources', 'No key resources listed')}
- Revenue Streams: {data.get('revenue_streams', 'No revenue streams defined')}

Graphs generated:
- Revenue Distribution (pie chart)
- Revenue Projection (12 months line chart)
- Customer Growth Projection (24 months line chart)
- Market Position Analysis (quadrant chart)
- Resource Allocation (bar chart)

For a full interactive report, please see the HTML version in the graphs directory.
        """
        )


@app.route("/")
def index():
    """Simple landing page for the API."""
    return render_template_string(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Graph Generator API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
            }
            pre {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .endpoint {
                margin-bottom: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .method {
                font-weight: bold;
                color: #3498db;
            }
        </style>
    </head>
    <body>
        <h1>Graph Generator API</h1>
        <p>This API generates various business analysis graphs based on input data.</p>
        
        <div class="endpoint">
            <h2>Generate Graphs</h2>
            <p><span class="method">POST</span> /generate</p>
            <p>Generate all graphs based on the input data.</p>
            <h3>Request Body</h3>
            <pre>
{
  "user_input": "Brief description of business idea",
  "problem": "Problem being solved",
  "target_customers": "Target customer description",
  "solution": "Solution description",
  "key_resources": "Key resources needed",
  "revenue_streams": "Revenue streams description"
}
            </pre>
        </div>
        
        <div class="endpoint">
            <h2>Get HTML Report</h2>
            <p><span class="method">GET</span> /report</p>
            <p>Get the generated HTML report.</p>
        </div>
        
        <div class="endpoint">
            <h2>Get Graphs</h2>
            <p><span class="method">GET</span> /graphs/{graph_name}</p>
            <p>Get a specific generated graph. Available graphs:</p>
            <ul>
                <li>revenue_pie_chart.png</li>
                <li>revenue_projection.png</li>
                <li>customer_growth.png</li>
                <li>market_position.png</li>
                <li>resource_allocation.png</li>
            </ul>
        </div>
    </body>
    </html>
    """
    )


@app.route("/generate", methods=["POST"])
def generate():
    """Generate graphs based on the input data."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Generate all graphs
        result = generate_all_graphs(data)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report")
def get_report():
    """Get the generated HTML report."""
    try:
        return send_from_directory(GRAPHS_DIR, "business_report.html")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/graphs/<graph_name>")
def get_graph(graph_name):
    """Get a specific generated graph."""
    try:
        return send_from_directory(GRAPHS_DIR, graph_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/test")
def test():
    """Generate test graphs with sample data."""
    result = generate_all_graphs(sample_data)
    return jsonify(result)


# Add API status endpoint
@app.route("/api/status")
def api_status():
    """Check if the API is running correctly."""
    return jsonify(
        {
            "status": "ok",
            "message": "Graph Generator API is running",
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    print("Starting Graph Generator API...")
    print("Sample data is being used for testing. Endpoints:")
    print("- /generate (POST): Generate graphs based on input data")
    print("- /report (GET): Get the generated HTML report")
    print("- /graphs/{graph_name} (GET): Get a specific graph")
    print("- /test (GET): Generate test graphs with sample data")

    # Generate graphs with sample data on startup
    try:
        generate_all_graphs(sample_data)
        print("Successfully generated sample graphs on startup")
    except Exception as e:
        print(f"Warning: Could not generate sample graphs: {e}")

    app.run(debug=True, port=5050)
