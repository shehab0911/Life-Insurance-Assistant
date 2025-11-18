
import os
import sys
import asyncio
from dotenv import load_dotenv


from .langgraph_agent import run_agent

load_dotenv()

async def main_loop():
    session_id = "cli_session_user"
    print("--- Insurance Assistant (CLI Mode) ---")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            
            text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not text:
            continue

        if text.lower() in ("exit", "quit"):
            print("Bye.")
            break

        if text.lower() == "reset":
            
            import uuid
            session_id = f"cli_{uuid.uuid4().hex[:8]}"
            print(f"Session reset (New ID: {session_id}).")
            continue

        try:
            
            answer = await run_agent(text, session_id)
        except Exception as e:
            answer = f"Error: {e}"

        print(f"Assistant: {answer}\n")
        sys.stdout.flush()

def main():
    
    asyncio.run(main_loop())

if __name__ == "__main__":
    
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    main()