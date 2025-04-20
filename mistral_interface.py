import subprocess
import os
import requests
import json
import time

LLAMA_SERVER_PATH = "D:/Downloads/llama-b5159-bin-win-avx2-x64/llama-server.exe"
SERVER_URL = "http://localhost:8080"

def start_server(model_path):
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Start the server with optimized settings for speed
    cmd = [
        LLAMA_SERVER_PATH,
        "-m", model_path,
        "--port", "8080",
        "-c", "1024",  # Reduced context window for faster processing
        "--no-warmup",  # Skip initial warmup
        "--threads", "4",  # Use all available threads
        "--batch-size", "512",  # Increased batch size for faster processing
        "--temp", "0.7",  # Slightly reduced temperature for faster, more focused responses
        "--repeat-penalty", "1.1"  # Slightly reduced penalty for faster generation
    ]
    
    print("Starting Mistral server (logging to logs/server.log)...")
    try:
        with open("logs/server.log", "w") as log_file:
            server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                creationflags=subprocess.CREATE_NO_WINDOW  # Hide console window on Windows
            )
        
        # Wait for server to start with a cleaner progress indicator
        print("Initializing model (this may take a minute)...")
        for i in range(60):  # Reduced timeout to 1 minute
            try:
                response = requests.get(f"{SERVER_URL}/health")
                if response.status_code == 200:
                    print("✓ Server is ready!")
                    return server_process
            except requests.exceptions.ConnectionError:
                if i % 5 == 0:  # Show progress every 5 seconds
                    print(".", end="", flush=True)
                time.sleep(1)
        
        # If we get here, the server didn't start
        with open("logs/server.log", "r") as log_file:
            error_log = log_file.read()
        raise Exception(f"Server failed to start after 1 minute. Server log:\n{error_log}")
        
    except Exception as e:
        raise Exception(f"Failed to start server: {str(e)}")

def ask_mistral(model_path, question, context):
    prompt = f"[INST] Use the following context to answer the question:\n\n{context}\n\nQuestion: {question} [/INST]"
    
    # Send request to the server with optimized parameters
    response = requests.post(
        f"{SERVER_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": 256,  # Reduced max tokens for faster responses
            "temperature": 0.7,
            "stop": ["</s>", "[INST]"],
            "top_k": 40,  # Added top_k for faster sampling
            "top_p": 0.9,  # Added top_p for faster sampling
            "repeat_penalty": 1.1  # Slightly reduced penalty
        }
    )
    
    if response.status_code == 200:
        return response.json()["content"].strip()
    else:
        print(f"Error from server: {response.text}")
        return "Error: Failed to get response from model"
