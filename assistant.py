from openai import OpenAI
import time
import json
import pandas as pd

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

client = OpenAI()

# Function that our server can call when LLM decides so and Run object switches to requires_action state.
def get_correlation(filename: str):
    df = pd.read_csv(filename)
    return df["Current (A)"].corr(df["Voltage (V)"])

# Create an assistant
my_assistant = client.beta.assistants.create(
    instructions="You are a personal data analist. You analyse tables that users provide with their files.",
    name="Data Analist",
    model="gpt-3.5-turbo-1106",

    tools=[{
            "type": "code_interpreter"
            }, 
            {
            "type": "function",
            "function": {
                  "name": "getCurrentWeather",
                  "description": "Get the weather in location",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                      "unit": {"type": "string", "enum": ["c", "f"]}
                    },
                    "required": ["location"]
                  }
                }
            }, 
            {
            "type": "function",
            "function": {
            "name": "get_correlation",
            "description": "Get the correlation between Current and Voltage in a csv table",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Provided CSV filename"},
                    },
                    "required": ["filename"]
                }
              } 
            }
        ]
)


# Create a user thread
thread = client.beta.threads.create()


# Upload a file with an "assistants" purpose
file = client.files.create(
  file=open("./data/EV_Battery_Data.csv", "rb"),
  purpose='assistants'
)


# Add a message to a thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Find the correlation between Voltage and Current in file 'data/EV_Battery_Data.csv'.", # 
    # file_ids=[file.id]
)


# Run the assistant (start it)
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=my_assistant.id,
    instructions="Help the user with his issue.",
)


messages = client.beta.threads.messages.list(thread_id=thread.id)
print(f"{BLUE}USER MESSAGE{RESET}: {messages.data[-1].content[0].text.value}")


# Get the status of a Run
run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
count = 0
while (run.status != "requires_action") and count < 10:
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print(f"{YELLOW}RUN STATUS{RESET}: {run.status}")
    time.sleep(1)
    count += 1


print(f"{BLUE}RUN OBJECT{RESET}: {run}")


print(f"{BLUE}SELECTED ACTION{RESET}: {run.required_action.submit_tool_outputs.tool_calls[0].function}")
function_name = run.required_action.submit_tool_outputs.tool_calls[0].function.name
call_id = run.required_action.submit_tool_outputs.tool_calls[0].id
extracted_filename = json.loads(run.required_action.submit_tool_outputs.tool_calls[0].function.arguments)["filename"]

if function_name == "get_correlation": # LLM indicates that we should call this function
    result = get_correlation(extracted_filename)

    run = client.beta.threads.runs.submit_tool_outputs( # Send results back
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=[
              {
                "tool_call_id": call_id,
                "output": str(result),
              }
            ]
    )


count = 0
while (run.status != "completed") and count < 10:
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    print(f"{YELLOW}RUN STATUS{RESET}: {run.status}")
    time.sleep(1)
    count += 1


messages = client.beta.threads.messages.list(thread_id=thread.id)
print(f"{BLUE}ASSISTANT MESSAGE{RESET}: {messages.data[0].content[0].text.value}")

