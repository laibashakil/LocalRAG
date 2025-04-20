import argparse
from utils.web_scraper import scrape_url
from utils.file_loader import load_text_file
from utils.embedder import Embedder
from utils.mistral_interface import start_server, ask_mistral
import atexit
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("\n" + "="*50)
    print("🧠 Local RAG System with Mistral")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="URL to scrape")
    parser.add_argument("--file", type=str, help="Text file to load")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Mistral GGUF model")
    args = parser.parse_args()

    print_header()

    if not args.url and not args.file:
        print("❌ Error: Provide either a --url or a --file")
        return

    # Load data
    if args.url:
        print(f"🌐 Scraping from URL: {args.url}")
        raw_text = scrape_url(args.url)
    else:
        print(f"📄 Loading from file: {args.file}")
        raw_text = load_text_file(args.file)

    # Initialize embedder + index text
    print("\n🔄 Initializing embedder...")
    embedder = Embedder()
    embedder.index_documents(raw_text)
    print("✓ Embedder ready")

    # Start the Mistral server
    server_process = start_server(args.model_path)
    
    # Register cleanup function to stop server on exit
    def cleanup():
        print("\n🛑 Stopping Mistral server...")
        server_process.terminate()
    atexit.register(cleanup)

    print("\n✨ System is ready! You can start asking questions.")
    print("📝 Type 'exit' to quit the program.\n")

    # Question loop
    while True:
        try:
            query = input("\n❓ Your question: ")
            if query.lower() == 'exit':
                break
            
            print("\n🔍 Finding relevant context...")
            context = embedder.retrieve_context(query)
            
            print("🤔 Generating response...")
            response = ask_mistral(args.model_path, query, context)
            print("\n🧠 Answer:", response.strip())
            print("\n" + "-"*50)
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
