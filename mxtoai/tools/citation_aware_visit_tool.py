"""
Citation-aware webpage visit tool that collects citations.
"""

import json
import logging
import re
from typing import ClassVar

from smolagents import Tool
from smolagents.default_tools import VisitWebpageTool

from mxtoai.request_context import RequestContext
from mxtoai.schemas import ToolOutputWithCitations

logger = logging.getLogger(__name__)


class CitationAwareVisitTool(Tool):
    """
    Visit webpage tool that automatically collects citations.
    Extends the default VisitWebpageTool to include citation tracking.
    """

    name = "visit_webpage_with_citations"
    description = (
        "Visit a webpage and extract its content while automatically tracking citations. "
        "This tool will add the visited URL to the citations collection and "
        "return the webpage content along with citation metadata."
    )
    inputs: ClassVar = {
        "url": {"type": "string", "description": "The URL of the webpage to visit."},
    }
    output_type = "object"

    def __init__(self, context: RequestContext):
        """Initialize the citation-aware visit tool."""
        super().__init__()
        self.context = context
        self.visit_tool = VisitWebpageTool()
        logger.debug("CitationAwareVisitTool initialized")

    def forward(self, url: str) -> str:
        """Visit a webpage and return content with citations."""
        try:
            logger.info(f"Visiting webpage: {url}")

            # Get the webpage content
            content = self.visit_tool.forward(url=url)

            # Extract title from content if possible
            title_match = (
                re.search(r"<title>(.*?)</title>", content, re.IGNORECASE)
                or re.search(r"<h1[^>]*>(.*?)</h1>", content, re.IGNORECASE)
                or re.search(r"# (.*?)$", content, re.MULTILINE)
            )

            title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else f"Webpage: {url}"

            # Add citation for this webpage - mark as visited
            citation_id = self.context.add_web_citation(url, title, visited=True)

            # Create structured output with citation
            result = ToolOutputWithCitations(
                content=f"**{title}** [#{citation_id}]\n\n{content}",
                metadata={"url": url, "title": title, "citation_id": citation_id, "content_length": len(content)},
            )

            logger.info(f"Successfully visited webpage and added citation [#{citation_id}]")
            return json.dumps(result.model_dump())

        except Exception as e:
            logger.error(f"Failed to visit webpage {url}: {e}")
            raise
