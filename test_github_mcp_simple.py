#!/usr/bin/env python3
"""
Simple GitHub MCP test using smolagents native support.
"""

import os
import dramatiq
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dramatiq.actor
def count_words2(url):
    import requests
    response = requests.get(url)
    count = len(response.text.split(" "))
    print(f"There are {count} words at {url!r}.")
    return count

@dramatiq.actor
def github_mcp_smolagents_native():
    """Test GitHub MCP server using Smolagents native MCPClient."""
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        return False
    from mcp import StdioServerParameters
    from smolagents.mcp_client import MCPClient

    # Configure GitHub MCP server parameters
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token}
    )

    # Use Smolagents native MCP client
    with MCPClient(server_params) as tools:
        for _tool in tools[:5]:  # Show first 5 tools
            print(f"Tool: {str(_tool)}")

        print("done with for loop")

    print("Finished testing GitHub MCP server with smolagents native client.")
    return True


@dramatiq.actor
def github_mcp_direct():
    """Test GitHub MCP server directly with smolagents."""
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        msg = "GITHUB_PERSONAL_ACCESS_TOKEN environment variable is not set."
        raise ValueError(msg)

    from mcp import StdioServerParameters
    from smolagents import ToolCollection

    # Configure GitHub MCP server parameters
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token}
    )

    # Use smolagents native MCP support
    with ToolCollection.from_mcp(server_params, trust_remote_code=True) as tool_collection:
        tools = list(tool_collection.tools)

        for _tool in tools:
            pass

        return True

@dramatiq.actor
def email_agent_with_github():
    """Test email agent with GitHub MCP integration."""
    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        return False

    from mxtoai.agents.email_agent import EmailAgent
    from mxtoai.models import ProcessingInstructions
    from mxtoai.schemas import EmailRequest

    # Create email agent
    agent = EmailAgent()

    # Create test email
    email = EmailRequest(
        subject="GitHub Repository Search",
        from_email="test@example.com",
        to="agent@mxtoai.com",
        textContent="Can you search for popular Python machine learning repositories on GitHub and give me a summary of the top 3?"
    )

    # Create processing instructions
    instructions = ProcessingInstructions(
        handle="research",
        aliases=["research", "github"],
        process_attachments=False,
        deep_research_mandatory=False
    )

    result = agent.process_email(email, instructions)
    return bool(result.email_content and result.email_content.text)


if __name__ == "__main__":

    # Test 1: Smolagents native MCP client
    test1_success = github_mcp_smolagents_native()

    # Test 2: Direct MCP connection (fallback)
    test2_success = github_mcp_direct()

    # Test 3: Email agent integration
    test3_success = email_agent_with_github()