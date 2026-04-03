"""Test tool calling with Gemma 4 via vLLM's OpenAI-compatible API."""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco, CA'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="google/gemma-4-E2B-it",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

message = response.choices[0].message
print("Response:", message)

if message.tool_calls:
    for tc in message.tool_calls:
        print(f"\nTool call: {tc.function.name}")
        print(f"Arguments: {tc.function.arguments}")
else:
    print("Content:", message.content)
