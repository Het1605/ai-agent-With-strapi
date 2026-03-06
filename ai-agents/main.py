# import asyncio
# from mcp.client.stdio import stdio_client, StdioServerParameters
# from mcp import ClientSession


# async def test_mcp():

#     server_params = StdioServerParameters(
#         command="npx",
#         args=["-y", "@bschauer/strapi-mcp-server"]
#     )

#     async with stdio_client(server_params) as (read, write):

#         async with ClientSession(read, write) as session:

#             await session.initialize()

#             tools = await session.list_tools()

#             print("\nAvailable MCP tools:\n")

#             for tool in tools:
#                 print("\nTool Name:", tool[0])
#                 print("Schema:", tool[1])


# if __name__ == "__main__":
#     asyncio.run(test_mcp())



import asyncio
import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import subprocess


from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY
)


# -----------------------------
# LLM: Extract schema
# -----------------------------
def extract_schema(user_query):

    prompt = f"""
                You convert user requests into a database schema.

                    User query:
                    {user_query}

                    Return ONLY valid JSON.

                    Return this structure:

                    {{
                    "collectionName": "table_name",
                    "fields": [
                        {{
                        "name": "field_name",
                        "type": "string | integer | boolean | date | datetime | email | relation",
                        "required": true/false,
                        "unique": true/false,
                        "default": value,
                        "relation": "manyToOne | oneToMany | manyToMany",
                        "target": "api::collection.collection"
                        }}
                    ]
                    }}

                    Rules:
                    - Include "required" if mentioned
                    - Include "unique" if mentioned
                    - Include "default" if mentioned
                    - Include relation fields if mentioned
                    - Do not omit properties if they exist

                    Return ONLY JSON.
                """

    response = llm.invoke(prompt)

    content = response.content.strip()

    # remove markdown if present
    content = content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)


# -----------------------------
# MCP Call
# -----------------------------
async def create_collection(schema):

    plural = schema["collectionName"]
    singular = plural[:-1]

    base_path = f"/strapi/src/api/{singular}/content-types/{singular}"
    os.makedirs(base_path, exist_ok=True)

    attributes = {}

    for field in schema["fields"]:

        attr = {
            "type": field["type"]
        }

        if "required" in field:
            attr["required"] = field["required"]

        if "unique" in field:
            attr["unique"] = field["unique"]

        if "default" in field:
            attr["default"] = field["default"]

        attributes[field["name"]] = attr

    schema_json = {
        "kind": "collectionType",
        "collectionName": plural,
        "info": {
            "singularName": singular,
            "pluralName": plural,
            "displayName": singular.capitalize()
        },
        "options": {
            "draftAndPublish": False
        },
        "attributes": attributes
    }

    schema_path = f"{base_path}/schema.json"

    with open(schema_path, "w") as f:
        json.dump(schema_json, f, indent=2)

    print("\nSchema file created correctly.\n")

    return {"status": "collection created"}

# -----------------------------
# Main Agent
# -----------------------------
async def main():

    print("\nType your query:")
    user_query = input("> ")

    print("\nUnderstanding request with LLM...\n")

    schema = extract_schema(user_query)

    print("Generated Schema:\n")
    print(json.dumps(schema, indent=2))

    print("\nCreating collection in Strapi...\n")

    result = await create_collection(schema)

    print("\nResult:\n")
    print(result)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())