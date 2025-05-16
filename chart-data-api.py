import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

    # If no valid revenue items were found, create some sample data
    if not values:
        values = [10, 50, 25]
        labels = ["Basic tier", "Premium tier", "Enterprise"]

    return values, labels


def generate_revenue_distribution_data(revenue_text):
    """Generate revenue distribution data for pie chart."""
    values, labels = parse_revenue_data(revenue_text)

    # Create data array for the pie chart
    data = []
    for i, (label, value) in enumerate(zip(labels, values)):
        data.append({"name": label, "value": value})

    return data


def generate_revenue_projection_data(revenue_text):
    """Generate a projection of revenue growth over time."""
    values, labels = parse_revenue_data(revenue_text)

    # Generate time series data (12 months)
    months = pd.date_range(start=datetime.now(), periods=12, freq="ME")
    month_labels = [month.strftime("%b %Y") for month in months]

    # Growth rates for each revenue stream (monthly)
    growth_rates = np.random.uniform(0.05, 0.15, len(values))

    # Create data array for the line chart
    data = []

    # Add data points for each month
    for j, month in enumerate(month_labels):
        month_data = {"name": month}

        # Calculate value for each revenue stream
        for i, (label, value) in enumerate(zip(labels, values)):
            # Apply compounding growth
            projected_value = value * (1 + growth_rates[i]) ** j
            month_data[label] = round(projected_value, 2)

        # Add total
        total = sum(
            value * (1 + growth_rates[i]) ** j
            for i, (_, value) in enumerate(zip(labels, values))
        )
        month_data["Total"] = round(total, 2)

        data.append(month_data)

    return {"data": data, "keys": labels + ["Total"]}


def generate_customer_growth_data(target_customers):
    """Generate a projection of customer growth over time."""
    # Parse target customers to estimate initial customer base
    words = len(target_customers.split())
    initial_customers = max(100, words * 10)  # Simple heuristic

    # Generate time series data (24 months)
    months = pd.date_range(start=datetime.now(), periods=24, freq="ME")
    month_labels = [month.strftime("%b %Y") for month in months]

    # Growth model parameters
    growth_rate = 0.1  # 10% monthly growth
    saturation = initial_customers * 10  # Maximum market size

    # S-curve growth model (logistic function)
    def logistic_growth(t, initial, saturation, growth_rate):
        return saturation / (
            1 + ((saturation / initial) - 1) * np.exp(-growth_rate * t)
        )

    # Generate customer growth projection
    data = []
    for j, month in enumerate(month_labels):
        customers = logistic_growth(j, initial_customers, saturation, growth_rate)
        data.append(
            {
                "name": month,
                "customers": round(customers, 0),
                "type": "Projected" if j < 12 else "Predicted",
            }
        )

    return data


def generate_market_position_data(problem, solution):
    """Generate market position data for quadrant chart."""
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

    # Create quadrant data
    quadrant_data = {
        "solution": {
            "x": round(uniqueness, 2),
            "y": round(value, 2),
            "name": "Your Solution",
        },
        "quadrants": [
            {"name": "Niche Players", "x": 2.5, "y": 7.5},
            {"name": "Market Leaders", "x": 7.5, "y": 7.5},
            {"name": "Low Priority", "x": 2.5, "y": 2.5},
            {"name": "Emerging Solutions", "x": 7.5, "y": 2.5},
        ],
    }

    return quadrant_data


def generate_resource_allocation_data(key_resources):
    """Generate resource allocation data for bar chart."""
    # Parse key resources
    resources = [r.strip() for r in key_resources.split(",")]

    # If no resources provided, use placeholder
    if not resources or resources == [""]:
        resources = ["Development", "Marketing", "Operations", "Sales", "Support"]

    # Generate random weights for resources
    weights = np.random.uniform(0.5, 1.0, len(resources))
    weights = weights / weights.sum() * 100  # Normalize to percentages

    # Sort for better visualization
    sorted_indices = np.argsort(weights)[::-1]
    sorted_resources = [resources[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]

    # Create data for bar chart
    data = []
    for name, value in zip(sorted_resources, sorted_weights):
        data.append({"name": name, "allocation": round(value, 1)})

    return data


def generate_all_chart_data(data):
    """Generate all chart data based on the input data."""
    try:
        # Generate data for each chart type
        revenue_distribution = generate_revenue_distribution_data(
            data.get("revenue_streams", "")
        )
        revenue_projection = generate_revenue_projection_data(
            data.get("revenue_streams", "")
        )
        customer_growth = generate_customer_growth_data(
            data.get("target_customers", "")
        )
        market_position = generate_market_position_data(
            data.get("problem", ""), data.get("solution", "")
        )
        resource_allocation = generate_resource_allocation_data(
            data.get("key_resources", "")
        )

        # Combine all data
        chart_data = {
            "revenue_distribution": revenue_distribution,
            "revenue_projection": revenue_projection,
            "customer_growth": customer_growth,
            "market_position": market_position,
            "resource_allocation": resource_allocation,
            "business_summary": {
                "problem": data.get("problem", "No problem description provided"),
                "target_customers": data.get(
                    "target_customers", "No target customers specified"
                ),
                "solution": data.get("solution", "No solution described"),
                "key_resources": data.get("key_resources", "No key resources listed"),
                "revenue_streams": data.get(
                    "revenue_streams", "No revenue streams defined"
                ),
            },
        }

        return chart_data
    except Exception as e:
        print(f"Error generating chart data: {str(e)}")
        return {"error": str(e)}


@app.route("/")
def index():
    """Simple landing page for the API."""
    return jsonify(
        {
            "message": "Chart Data API is running. Send a POST request to /generate-data to get chart data."
        }
    )


@app.route("/generate-data", methods=["POST"])
def generate_data():
    """Generate chart data based on the input data."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Generate all chart data
        result = generate_all_chart_data(data)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/test-data")
def test_data():
    """Generate test chart data with sample data."""
    result = generate_all_chart_data(sample_data)
    return jsonify(result)


# Add API status endpoint
@app.route("/api/status")
def api_status():
    """Check if the API is running correctly."""
    return jsonify(
        {
            "status": "ok",
            "message": "Chart Data API is running",
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    print("Starting Chart Data API...")
    print("Sample data is being used for testing. Endpoints:")
    print("- /generate-data (POST): Generate chart data based on input")
    print("- /test-data (GET): Get test chart data with sample input")
    print("- /api/status (GET): Check API status")

    app.run(debug=True, port=5050)
