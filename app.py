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
FRONTEND_URL = "https://exoplanet-frontend.vercel.app" 

app = Flask(__name__)
# Initialize CORS and allow the Vercel domain to communicate with this API
CORS(app, resources={r"/*": {"origins": FRONTEND_URL}})


# --- LLM API Configuration (Leave as blank string for the Canvas environment) ---
GEMINI_API_KEY = ""
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

def generate_llm_summary(df_summary):
    """Generates a contextual summary using the Gemini API with exponential backoff."""
    
    # Use to_markdown for a cleaner, more readable input for the LLM
    summary_text = df_summary.to_markdown(index=True, floatfmt=".2f")
    
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
        # Using Google Search grounding tool to provide context and grounding for the analysis
        "tools": [{"google_search": {}}], 
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    headers = {'Content-Type': 'application/json'}
    
    # Use exponential backoff for robustness
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Note: The API key is added to the URL for the Canvas environment
            response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                                     headers=headers, 
                                     data=json.dumps(payload))
            response.raise_for_status() 
            
            result = response.json()
            
            # Check for generated text
            if result and 'candidates' in result and result['candidates']:
                text_part = result['candidates'][0]['content']['parts'][0]['text']
                
                # Check for sources/citations from grounding
                sources = []
                grounding_metadata = result['candidates'][0].get('groundingMetadata')
                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources = [
                        { "uri": attr['web']['uri'], "title": attr['web']['title'] }
                        for attr in grounding_metadata['groundingAttributions']
                        if attr.get('web') and attr['web'].get('uri')
                    ]

                return {"text": text_part, "sources": sources}
            
            # Fallback if text generation failed
            return {"text": "Analysis successful, but LLM summary failed to generate text.", "sources": []}

        except requests.exceptions.RequestException as e:
            # Handle specific status codes for retry
            if attempt < max_retries - 1 and ('response' in locals() and response.status_code in [429, 500, 503] or 'response' not in locals()):
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            return {"text": f"Error during LLM API call: {e}", "sources": []}
        except Exception as e:
            return {"text": f"An unexpected error occurred during LLM processing: {e}", "sources": []}
    
    return {"text": "LLM API call failed after multiple retries.", "sources": []}


@app.route('/', methods=['GET'])
def home():
    """A friendly route for the root URL."""
    return jsonify({
        "message": "Welcome to the Exoplanet Data Analysis API!",
        "analysis_endpoint": "/upload (POST a CSV file)",
        "status": "Ready"
    }), 200


@app.route('/upload', methods=['POST'])
def upload_and_analyze():
    """Handles CSV file upload, performs analysis, and returns results."""
    
    # 1. Check for file in request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Please upload a CSV file."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "File must be a CSV format (e.g., data.csv)."}), 400

    try:
        # 2. Read CSV file into a Pandas DataFrame
        df = pd.read_csv(file)
        
        # 3. Data Cleaning and Preparation
        required_cols = ['Planet Mass (M_Jup)', 'Orbit Period (days)', 'Detection Method']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Missing required column: '{col}'. Ensure CSV columns match: {', '.join(required_cols)}"}), 400
        
        # Convert numerical columns to numeric, coercing errors to NaN
        for col in ['Planet Mass (M_Jup)', 'Orbit Period (days)']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values in critical columns
        df.dropna(subset=['Planet Mass (M_Jup)', 'Orbit Period (days)', 'Detection Method'], inplace=True)
        
        # If no data is left after cleaning, return an informative error
        if df.empty:
            return jsonify({"error": "No valid data rows remaining after cleaning (check for missing or non-numeric values in critical columns)."}), 400


        # 4. Perform Statistical Analysis
        stats_df = df[['Planet Mass (M_Jup)', 'Orbit Period (days)']].describe().reset_index()
        stats_data = stats_df.to_dict(orient='records')
        
        # 5. Generate Matplotlib Plot (Planet Mass vs. Orbit Period by Detection Method)
        x_col = 'Planet Mass (M_Jup)'
        y_col = 'Orbit Period (days)'
        
        plt.figure(figsize=(12, 8))
        
        detection_methods = df['Detection Method'].unique()
        
        # Limit to top 8 methods for readability in the legend
        top_methods = df['Detection Method'].value_counts().nlargest(8).index.tolist()
        
        # Apply log scale for better visualization of orbital periods
        plt.yscale('log') 
        
        for method in top_methods:
            subset = df[df['Detection Method'] == method]
            plt.scatter(
                subset[x_col], 
                subset[y_col], 
                label=method, 
                alpha=0.6,
                edgecolors='w', 
                s=50
            )
            
        plt.xlabel(x_col, fontsize=14)
        plt.ylabel(f"{y_col} (log scale)", fontsize=14)
        plt.title('Exoplanet Mass vs. Orbital Period by Detection Method', fontsize=16)
        plt.legend(title='Detection Method', fontsize=10, loc='lower right')
        plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout() # Ensures labels don't get cut off
        
        # Convert Matplotlib plot to base64 image string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plot_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()

        # 6. Generate LLM Summary
        llm_analysis = generate_llm_summary(stats_df)
        
        # 7. Return all results
        return jsonify({
            "status": "success",
            "message": f"Analysis complete for {len(df)} valid data points.",
            "stats": stats_data,
            "plot_image": plot_base64,
            "summary_text": llm_analysis.get('text'),
            "summary_sources": llm_analysis.get('sources')
        }), 200

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Server-side processing error: {e}")
        return jsonify({"error": f"Internal server error during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    # When running locally, Flask is used.
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
