from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import base64
import os
import requests
import json
import time

# --- Configuration ---
# Your Vercel frontend URL is the *only* allowed origin for CORS requests.
# This fixes the Network Error when the frontend tries to contact the backend.
FRONTEND_URL = "https://exoplanet-frontend.vercel.app" 

app = Flask(__name__)
# Initialize CORS and allow the Vercel domain to communicate with this API
CORS(app, resources={r"/*": {"origins": FRONTEND_URL}})


# --- LLM API Configuration (Leave as blank string) ---
# NOTE: The Canvas environment automatically provides the API key at runtime.
GEMINI_API_KEY = ""
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def generate_llm_summary(df_summary):
    """Generates a contextual summary using the Gemini API."""
    
    # Format the DataFrame summary into a readable string
    summary_text = df_summary.to_string()
    
    system_prompt = (
        "You are an expert astrophysical data analyst. Analyze the provided Pandas DataFrame "
        "summary statistics (count, mean, std, min, max, quartiles) for exoplanet data. "
        "Focus on the numerical columns like 'Planet Mass (M_Jup)' and 'Orbit Period (days)'. "
        "Provide a concise, single-paragraph summary of the key findings, such as the typical "
        "range of planet mass, the longest/shortest orbit period, and any notable averages. "
        "Use a sophisticated and informative tone."
    )
    
    user_query = f"Analyze this exoplanet data summary statistics and provide a concise analysis:\n\n{summary_text}"
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    headers = {'Content-Type': 'application/json'}
    
    # Use exponential backoff for robustness
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                                     headers=headers, 
                                     data=json.dumps(payload))
            response.raise_for_status() 
            
            result = response.json()
            # Extract the text from the response
            if result and 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
            return "Analysis successful, but LLM summary failed to generate text."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1 and (response.status_code in [429, 500, 503] if 'response' in locals() else True):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            return f"Error during LLM API call: {e}"
        except Exception as e:
            return f"An unexpected error occurred during LLM processing: {e}"
    return "LLM API call failed after multiple retries."


@app.route('/')
def hello_world():
    """Simple health check endpoint."""
    return jsonify({"message": "Exoplanet Analysis API is running!"})


@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    """Handles CSV file upload, performs analysis, and returns results."""
    
    # 1. Check for file in request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be a CSV format"}), 400

    try:
        # 2. Read CSV file into a Pandas DataFrame
        df = pd.read_csv(file)
        
        # 3. Data Cleaning and Preparation
        required_cols = ['Planet Mass (M_Jup)', 'Orbit Period (days)', 'Detection Method']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing required column: '{col}'"}), 400
        
        # Convert numerical columns to numeric, coercing errors to NaN
        for col in ['Planet Mass (M_Jup)', 'Orbit Period (days)']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values in critical columns
        df.dropna(subset=['Planet Mass (M_Jup)', 'Orbit Period (days)', 'Detection Method'], inplace=True)

        # 4. Perform Statistical Analysis
        stats_df = df[['Planet Mass (M_Jup)', 'Orbit Period (days)']].describe().reset_index()
        stats_data = stats_df.to_dict(orient='records')
        
        # 5. Generate Matplotlib Plot (Planet Mass vs. Orbit Period by Detection Method)
        
        # Define the numerical columns for the scatter plot
        x_col = 'Planet Mass (M_Jup)'
        y_col = 'Orbit Period (days)'
        
        plt.figure(figsize=(12, 8))
        
        # Plot each detection method separately for labeling
        detection_methods = df['Detection Method'].unique()
        
        # Use log scale for Orbit Period for better visualization due to large range
        plt.yscale('log')
        
        for method in detection_methods:
            subset = df[df['Detection Method'] == method]
            plt.scatter(
                subset[x_col], 
                subset[y_col], 
                label=method, 
                alpha=0.6,
                edgecolors='w', 
                s=50 # size of markers
            )
            
        plt.xlabel(x_col, fontsize=14)
        plt.ylabel(f"{y_col} (log scale)", fontsize=14)
        plt.title('Exoplanet Mass vs. Orbital Period', fontsize=16)
        plt.legend(title='Detection Method', fontsize=10)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        
        # Convert Matplotlib plot to base64 image string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close() # Close the plot to free memory

        # 6. Generate LLM Summary
        llm_summary = generate_llm_summary(stats_df)
        
        # 7. Return all results
        return jsonify({
            "status": "success",
            "message": "Analysis complete",
            "stats": stats_data,
            "plot_image": plot_base64,
            "summary": llm_summary
        }), 200

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Server-side processing error: {e}")
        return jsonify({"error": f"Internal server error during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    # When running locally, you can change the host to allow external connections 
    # and set a higher port number if needed.
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
