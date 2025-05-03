import asyncio
from openai import OpenAI

client = OpenAI()

async def main():
    # Your agent/assistant logic here
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

# This runs the async function
asyncio.run(main())