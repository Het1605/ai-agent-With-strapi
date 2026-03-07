import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession


async def test_mcp():

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@bschauer/strapi-mcp-server"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\nAvailable MCP tools:\n")

            for tool in tools:
                print("\nTool Name:", tool[0])
                print("Schema:", tool[1])


if __name__ == "__main__":
    asyncio.run(test_mcp())


