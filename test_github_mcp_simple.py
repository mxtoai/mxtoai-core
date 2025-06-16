#!/usr/bin/env python3
"""
Simple GitHub MCP test using smolagents native support.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_github_mcp_smolagents_native():
    """Test GitHub MCP server using Smolagents native MCPClient."""

    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        print("❌ GITHUB_PERSONAL_ACCESS_TOKEN not found")
        print("   Set it with: export GITHUB_PERSONAL_ACCESS_TOKEN='your_token_here'")
        return False

    print("🔄 Testing GitHub MCP server with Smolagents native client...")

    try:
        from smolagents.mcp_client import MCPClient
        from mcp import StdioServerParameters

        # Configure GitHub MCP server parameters
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token}
        )

        # Use Smolagents native MCP client
        with MCPClient(server_params) as tools:
            print(f"✅ Successfully connected! Found {len(tools)} tools:")
            for tool in tools[:5]:  # Show first 5 tools
                print(f"  - {tool.name}: {tool.description}")

            if len(tools) > 5:
                print(f"  ... and {len(tools) - 5} more tools")

            return True

    except Exception as e:
        print(f"❌ Failed to connect to GitHub MCP server: {e}")
        return False

def test_github_mcp_direct():
    """Test GitHub MCP server directly with smolagents."""

    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        print("❌ GITHUB_PERSONAL_ACCESS_TOKEN not found")
        print("   Set it with: export GITHUB_PERSONAL_ACCESS_TOKEN='your_token_here'")
        return False

    print("🔄 Testing GitHub MCP server connection...")

    try:
        from smolagents import ToolCollection
        from mcp import StdioServerParameters

        # Configure GitHub MCP server parameters
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token}
        )

        # Use smolagents native MCP support
        with ToolCollection.from_mcp(server_params, trust_remote_code=True) as tool_collection:
            tools = list(tool_collection.tools)

            print(f"✅ Successfully connected! Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")

            return True

    except Exception as e:
        print(f"❌ Failed to connect to GitHub MCP server: {e}")
        return False

def test_email_agent_with_github():
    """Test email agent with GitHub MCP integration."""

    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not github_token:
        print("❌ GITHUB_PERSONAL_ACCESS_TOKEN not found for email agent test")
        return False

    print("\n🔄 Testing Email Agent with GitHub MCP...")

    try:
        from mxtoai.agents.email_agent import EmailAgent
        from mxtoai.schemas import EmailRequest
        from mxtoai.models import ProcessingInstructions

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

        print("🔄 Processing email with GitHub MCP tools...")
        result = agent.process_email(email, instructions)

        if result.email_content and result.email_content.text:
            print("✅ Email processed successfully!")
            print("📧 Response preview:")
            print(result.email_content.text[:500] + "..." if len(result.email_content.text) > 500 else result.email_content.text)
            return True
        else:
            print("❌ Email processing failed - no response generated")
            return False

    except Exception as e:
        print(f"❌ Email agent test failed: {e}")
        return False

def test_worker_simulation():
    """Test MCP loading in a simulated worker environment."""
    print("\n🔄 Testing MCP in simulated worker environment...")

    # Set environment variables to simulate dramatiq worker
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/test-dramatiq"

    try:
        from mxtoai.agents.email_agent import EmailAgent

        # Create agent (this will trigger MCP loading)
        agent = EmailAgent()

        # Try to get MCP tools
        mcp_tools = agent._get_mcp_tools_for_request()

        print(f"✅ Successfully loaded {len(mcp_tools)} MCP tools in simulated worker environment")
        return True

    except Exception as e:
        print(f"❌ Worker simulation test failed: {e}")
        return False
    finally:
        # Clean up environment
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            del os.environ["PROMETHEUS_MULTIPROC_DIR"]

if __name__ == "__main__":
    print("🧪 GitHub MCP Integration Tests")
    print("=" * 40)

    # Test 1: Smolagents native MCP client
    test1_success = test_github_mcp_smolagents_native()

    # Test 2: Direct MCP connection (fallback)
    test2_success = test_github_mcp_direct()

    # Test 3: Email agent integration
    test3_success = test_email_agent_with_github()

    # Test 4: Worker environment simulation
    test4_success = test_worker_simulation()

    print("\n📊 Test Results:")
    print(f"  Smolagents Native MCP Client: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"  Direct MCP Connection: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"  Email Agent Integration: {'✅ PASS' if test3_success else '❌ FAIL'}")
    print(f"  Worker Environment Simulation: {'✅ PASS' if test4_success else '❌ FAIL'}")

    if test1_success or test2_success:
        print("\n🎉 MCP connection successful! GitHub integration is working.")
        if test3_success:
            print("🎉 Email agent integration is also working.")
        if test4_success:
            print("🎉 Worker environment compatibility confirmed.")
        exit(0)
    else:
        print("\n❌ MCP connection failed. Check the logs above.")
        exit(1)