
import os
import httpx
import asyncio
from dotenv import load_dotenv

# Try loading from backend explicitly since we run from parent sometimes
load_dotenv("backend/.env")
# Also try current dir
load_dotenv(".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

async def list_models():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found.")
        return

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("Models containing 'scout':")
                for model in data['data']:
                    if "scout" in model['id'].lower():
                        print(f"MODEL: {model['id']}")
            else:
                print(f"Error Body: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(list_models())
