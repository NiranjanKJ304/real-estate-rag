# Real Estate AI Assistant

## Overview
The **Real Estate AI Assistant** is an intelligent web application built to assist users in exploring real estate options in Chennai, India. Powered by advanced AI models (Google Gemini) and FAISS for vector search, this tool allows users to upload property data (CSV/Excel) and guidelines (PDF), ask natural language queries, and receive detailed responses with market insights. The application features a dynamic, futuristic UI using Flask, HTML, JavaScript, and Tailwind CSS, with chat history displayed in the sidebar and query-response pairs in the main content area.

- **Developed by**: Niranjan KJ
- **Date**: July 29 - 31, 2025

## Features
- **Data Upload**: Upload property data (CSV/Excel) and guidelines (PDF) for processing.
- **Intelligent Search**: Uses FAISS index with Google GenerativeAI embeddings for semantic search.
- **Chat Interface**: Ask questions (e.g., "Show me 3BHK properties under ₹1 crore") and get detailed responses.
- **Quick Actions**: Predefined queries for budget properties, popular locations, and market trends.
- **Chat History**: View all past queries and responses in the sidebar.
- **Futuristic UI**: Dynamic design with neon gradients, glow effects, and responsive layout.

## System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for large datasets)
- **Internet**: Required for API calls and CSS/JS CDNs

## Installation

### Prerequisites
1. Install Python 3.9+ from [python.org](https://www.python.org/downloads/).
2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv

Activate the virtual environment:

Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate



Dependencies
Install the required Python packages:
bashpip install flask pandas==2.2.3 PyMuPDF==1.24.10 faiss-cpu==1.9.0 numpy==1.26.4 langchain-google-genai==2.0.1 langchain-core==0.3.8 python-dotenv
Google API Key

Obtain a Google API key from Google Cloud Console.
Set the API key as an environment variable:

Windows: set GOOGLE_API_KEY=your_api_key_here
macOS/Linux: export GOOGLE_API_KEY=your_api_key_here


Alternatively, create a .env file in the project directory with:
textGOOGLE_API_KEY=your_api_key_here
and ensure python-dotenv is installed.

Project Structure

app.py: Main application file containing Flask backend, HTML template, and logic.
README.md: This file.
data/: Directory for uploaded files (created automatically).
venv/: Virtual environment (if created).

Running the Application

Navigate to the project directory:
bashcd C:\Users\navan\OneDrive\Desktop\real_estate_ai_assistant

Activate the virtual environment (if not already active):

Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate


Run the application:
bashpython app.py

Open your browser and go to http://127.0.0.1:5000.

If port 5000 is in use, modify app.run(debug=True, port=5000) in app.py to a different port (e.g., port=5001).



Usage
Uploading Data

In the sidebar, upload a CSV/Excel file with property data (columns: Property ID, Project Name, Location, etc.).
Upload a PDF file with guidelines (optional).
Click "Load and Process Data" to build the search index. Progress messages appear in the sidebar.

Querying the Assistant

In the main content area, enter a query (e.g., "Show me 3BHK properties under ₹1 crore in Chennai") in the input field.
Click "Ask" to get a response. The query appears at the top, and the response below it.
View the full chat history in the sidebar.

Quick Actions

Budget Properties: Click "Show Budget Properties" for properties under ₹1 crore.
Popular Locations: Click "Popular Locations" for top investment areas.
Market Insights: Click "Market Insights" for trends in Chennai.

Clearing History

Click "Clear Chat History" to reset the sidebar chat log.

Sample Data
CSV Format
Create a properties.csv file:
csvProperty ID,Project Name,Location,Address,Status,Type,BHK,Size (sq.ft.),Start Price,Price/sq.ft,Amenities,Nearby,Furnishing,Contact Person,Contact,Offers
1,Elite Towers,Chennai,123 Main St,Available,Apartment,3BHK,1200,9500000,7917,Pool,Gym,Unfurnished,John Doe,1234567890,None
PDF Format
Create a guidelines.pdf with text (e.g., "All properties must comply with local zoning laws.").
Troubleshooting

API Key Issues: If "Gemini client not initialized" appears, verify your API key and environment variable setup.
Port Conflicts: If http://127.0.0.1:5000 is unavailable, change the port in app.py or terminate the conflicting process.
Dependency Errors: Reinstall missing packages:
bashpip install -r requirements.txt
(Create requirements.txt with listed dependencies if needed.)
UI Not Loading: Ensure an internet connection for Tailwind CSS and Google Fonts CDNs. Check browser console (F12) for errors.
File Upload Fails: Use valid CSV/Excel or PDF files. Test with small sample files.

Google for the GenerativeAI API.

Contact
For support, email: niranjan.kj2022ai-ds@sece.ac.in

### Ready-to-Download File
Since I can't host files directly, follow these steps to create a downloadable `.zip` file:

1. **Create the Files**:
   - Save the `README.md` content above as `README.md`.
   - Save the latest `app.py` (from the previous response) as `app.py` in the same directory.

2. **Create a ZIP File**:
   - On Windows:
     - Right-click the folder containing `README.md` and `app.py`.
     - Select "Send to" > "Compressed (zipped) folder".
     - Name it `real_estate_ai_assistant.zip`.
   - On macOS/Linux:
     ```bash
     zip -r real_estate_ai_assistant.zip README.md app.py

Download:

The resulting real_estate_ai_assistant.zip is ready to download from your local machine.
Share it via a file-sharing service (e.g., Google Drive, Dropbox) if needed.
