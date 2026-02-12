"""
Interactive CLI Chat with Zoho Books Agent
Ask questions and get answers in the terminal
"""

from zoho_books_all_in_one import ZohoBooksAgent

def main():
    print("=" * 60)
    print("🤖 Zoho Books Agent - Interactive Chat")
    print("=" * 60)
    print("Type your questions and press Enter.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    # Initialize agent
    try:
        agent = ZohoBooksAgent()
        print("✅ Agent initialized successfully!\n")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Chat loop
    history = []
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Run agent
            print("🤔 Thinking...", end="\r")
            context = {"history": history}
            result = agent.run(user_input, context)
            
            # Display response
            print(f"Agent: {result.text}\n")
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result.text})
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
