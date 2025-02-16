import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve variables
claude_api_key = os.getenv("CLAUDE_API_KEY")
print(f"API Key present: {'CLAUDE_API_KEY' in os.environ}")


if not claude_api_key:
    raise ValueError("CLAUDE_API_KEY environment variable not set.")
