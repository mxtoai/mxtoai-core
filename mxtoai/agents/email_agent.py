import ast
import json
import os
import re
from datetime import datetime
from typing import Any, Optional, Union
import threading

from dotenv import load_dotenv

# Update imports to use proper classes from smolagents
from smolagents import Tool, ToolCallingAgent

# Add imports for the new default tools
from smolagents.default_tools import (
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from sqlmodel import select

from mxtoai._logging import get_logger, get_smolagents_console
from mxtoai.config import SCHEDULED_TASKS_MAX_PER_EMAIL
from mxtoai.db import init_db_connection
from mxtoai.models.models import Tasks
from mxtoai.prompts.base_prompts import (
    MARKDOWN_STYLE_GUIDE,
    RESEARCH_GUIDELINES,
    RESPONSE_GUIDELINES,
    SECURITY_GUIDELINES,
)
from mxtoai.prompts.template_prompts import (
    SCHEDULED_TASK_CONTEXT_TEMPLATE,
    SCHEDULED_TASK_DISTILLED_INSTRUCTIONS_TEMPLATE,
    SCHEDULED_TASK_ERROR_TEMPLATE,
    SCHEDULED_TASK_NOT_FOUND_TEMPLATE,
)
from mxtoai.routed_litellm_model import RoutedLiteLLMModel
from mxtoai.schemas import (
    AgentResearchMetadata,
    AgentResearchOutput,
    AttachmentsProcessingResult,
    CalendarResult,
    DetailedEmailProcessingResult,
    EmailAttachment,
    EmailContentDetails,
    EmailRequest,
    EmailSentStatus,
    PDFExportResult,
    ProcessedAttachmentDetail,
    ProcessingError,
    ProcessingInstructions,
    ProcessingMetadata,
)
from mxtoai.scripts.report_formatter import ReportFormatter
from mxtoai.scripts.visual_qa import azure_visualizer
from mxtoai.tools.attachment_processing_tool import AttachmentProcessingTool
from mxtoai.tools.deep_research_tool import DeepResearchTool
from mxtoai.tools.delete_scheduled_tasks_tool import DeleteScheduledTasksTool
from mxtoai.tools.external_data.linkedin import initialize_linkedin_data_api_tool, initialize_linkedin_fresh_tool
from mxtoai.tools.meeting_tool import MeetingTool
from mxtoai.tools.pdf_export_tool import PDFExportTool
from mxtoai.tools.scheduled_tasks_tool import ScheduledTasksTool

# Import the web search tools
from mxtoai.tools.web_search import BraveSearchTool, DDGSearchTool, GoogleSearchTool

# Load environment variables
load_dotenv(override=True)

# Configure logger
logger = get_logger("email_agent")

# Define allowed imports for PythonInterpreterTool
ALLOWED_PYTHON_IMPORTS = [
    "datetime",
    "pytz",
    "math",
    "json",
    "re",
    "time",
    "collections",
    "itertools",
    "xml.etree.ElementTree",
    "csv",
    "urllib.parse",
]


class EmailAgent:
    """
    Email processing agent that can summarize, reply to, and research information for emails.
    """

    _mcp_failed_in_worker = False  # Class-level flag to track MCP failures in workers
    _mcp_tools_cache = None  # Class-level cache for MCP tools
    _mcp_cache_lock = threading.Lock()  # Thread-safe cache access

    def __init__(
        self, email_request: EmailRequest, attachment_dir: str = "email_attachments", verbose: bool = False, enable_deep_research: bool = False
    ):
        """
        Initialize the email agent with tools for different operations.

        Args:
            attachment_dir: Directory to store email attachments
            verbose: Whether to enable verbose logging
            enable_deep_research: Whether to enable Jina AI deep research functionality (uses API tokens)

        """
        # Set up logging
        if verbose:
            # Consider configuring logging level via environment variables or central logging setup
            logger.debug("Verbose logging potentially enabled (actual level depends on logger config).")

        self.attachment_dir = attachment_dir
        os.makedirs(self.attachment_dir, exist_ok=True)

        self.email_request = email_request
        self.attachment_tool = AttachmentProcessingTool()
        self.report_formatter = ReportFormatter()
        self.meeting_tool = MeetingTool()
        self.visit_webpage_tool = VisitWebpageTool()
        self.python_tool = PythonInterpreterTool(authorized_imports=ALLOWED_PYTHON_IMPORTS)
        self.wikipedia_search_tool = WikipediaSearchTool()
        self.pdf_export_tool = PDFExportTool()

        # Initialize scheduled tasks tool with call counter wrapper
        self.scheduled_tasks_tool = self._create_limited_scheduled_tasks_tool(email_request)
        self.delete_scheduled_tasks_tool = DeleteScheduledTasksTool(email_request=email_request)

        # Initialize independent search tools
        self.search_tools = self._initialize_independent_search_tools()
        self.research_tool = self._initialize_deep_research_tool(enable_deep_research)

        self.available_tools: list[Tool] = [
            self.attachment_tool,
            self.meeting_tool,
            self.visit_webpage_tool,
            self.python_tool,
            self.wikipedia_search_tool,
            self.pdf_export_tool,
            self.scheduled_tasks_tool,
            self.delete_scheduled_tasks_tool,
            azure_visualizer,
        ]

        # Add all available search tools
        self.available_tools.extend(self.search_tools)

        if self.research_tool:
            self.available_tools.append(self.research_tool)

        # Add LinkedIn tools
        linkedin_fresh_tool = initialize_linkedin_fresh_tool()
        if linkedin_fresh_tool:
            self.available_tools.append(linkedin_fresh_tool)

        linkedin_data_api_tool = initialize_linkedin_data_api_tool()
        if linkedin_data_api_tool:
            self.available_tools.append(linkedin_data_api_tool)

        logger.info(f"Agent tools initialized: {[tool.name for tool in self.available_tools]}")
        self._init_agent()
        logger.info("Email agent initialized successfully")

    def _init_agent(self):
        """Initialize the smolagents ToolCallingAgent."""
        # Initialize the routed model with the default model group
        self.routed_model = RoutedLiteLLMModel()

        # Create agent
        self.agent = ToolCallingAgent(
            model=self.routed_model,
            tools=self.available_tools,
            max_steps=12,
            verbosity_level=2,  # Increased back to 2 to capture detailed Rich console output
            planning_interval=4,
            name="mxtoai_email_processing_agent",
            description="I'm MXtoAI agent - an intelligent email processing agent that automates email-driven tasks and workflows. I can analyze emails, generate professional summaries and replies, conduct comprehensive research using web search and external APIs, process attachments (documents, images, PDFs), extract and create calendar events, export content to PDF, and execute code for data analysis. I maintain professional communication standards while providing accurate, well-researched responses tailored to your specific email handling requirements.",
            provide_run_summary=True,
        )

        # Set up integrated Rich console that feeds into loguru/logfire pipeline
        # This captures smolagents verbose output and integrates it with our unified logging
        smolagents_console = get_smolagents_console()

        # Override agent's console with our loguru-integrated console
        if hasattr(self.agent, "logger") and hasattr(self.agent.logger, "console"):
            self.agent.logger.console = smolagents_console
        if (
            hasattr(self.agent, "monitor")
            and hasattr(self.agent.monitor, "logger")
            and hasattr(self.agent.monitor.logger, "console")
        ):
            self.agent.monitor.logger.console = smolagents_console

        logger.debug("Agent initialized with routed model configuration, loguru-integrated Rich console")

    def _initialize_independent_search_tools(self) -> list[Tool]:
        """
        Initialize independent search tools for DDG, Brave, and Google.
        The agent will be able to choose which search engine to use based on cost and quality needs.

        Returns:
            list[Tool]: List of available search tools.

        """
        search_tools = []

        # DDG Search - Always available (free)
        ddg_tool = DDGSearchTool(max_results=10)
        search_tools.append(ddg_tool)
        logger.debug("Initialized DDG search tool (free, first choice)")

        # Brave Search - Available if API key is configured
        if os.getenv("BRAVE_SEARCH_API_KEY"):
            brave_tool = BraveSearchTool(max_results=5)
            search_tools.append(brave_tool)
            logger.debug("Initialized Brave search tool (moderate cost, better quality)")
        else:
            logger.warning("BRAVE_SEARCH_API_KEY not found. Brave search tool not initialized.")

        # Google Search - Available if API keys are configured
        if os.getenv("SERPAPI_API_KEY") or os.getenv("SERPER_API_KEY"):
            google_tool = GoogleSearchTool()
            search_tools.append(google_tool)
            logger.debug("Initialized Google search tool (premium cost, highest quality)")
        else:
            logger.warning("No Google Search API keys found. Google search tool not initialized.")

        logger.info(f"Initialized {len(search_tools)} independent search tools: {[tool.name for tool in search_tools]}")
        return search_tools

    def _get_required_actions(self, mode: str) -> list[str]:
        """
        Get list of required actions based on mode.

        Args:
            mode: The mode of operation (e.g., "summary", "reply", "research", "full")

        Returns:
            List[str]: List of actions to be performed by the agent

        """
        actions = []
        if mode in ["summary", "full"]:
            actions.append("Generate summary")
        if mode in ["reply", "full"]:
            actions.append("Generate reply")
        if mode in ["research", "full"]:
            actions.append("Conduct research")
        return actions

    def _initialize_deep_research_tool(self, enable_deep_research: bool) -> Optional[DeepResearchTool]:
        """
        Initializes the DeepResearchTool if API key is available.

        Args:
            enable_deep_research: Flag to enable deep research functionality

        Returns:
            Optional[DeepResearchTool]: Initialized DeepResearchTool instance or None if API key is not found

        """
        research_tool: Optional[DeepResearchTool] = None
        if os.getenv("JINA_API_KEY"):
            research_tool = DeepResearchTool()
            if enable_deep_research:
                # Assuming DeepResearchTool is enabled by its presence and API key.
                # If specific enabling logic is needed in DeepResearchTool, it should be called here.
                logger.debug(
                    "DeepResearchTool instance created; deep research functionality is active if enable_deep_research is true."
                )
            else:
                logger.debug(
                    "DeepResearchTool instance created, but deep research is not explicitly enabled via agent config (enable_deep_research=False). Tool may operate in a basic mode or not be used by agent logic if dependent on this flag."
                )
        else:
            logger.info("JINA_API_KEY not found. DeepResearchTool not initialized.")
        return research_tool

    def _create_email_context(self, email_request: EmailRequest, attachment_details=None) -> str:
        """
        Generate context information from the email request.

        Args:
            email_request: EmailRequest instance containing email data
            attachment_details: List of formatted attachment details

        Returns:
            str: The context information for the agent

        """
        recipients = ", ".join(email_request.recipients) if email_request.recipients else "N/A"
        attachments_info = (
            f"Available Attachments:\n{chr(10).join(attachment_details)}"
            if attachment_details
            else "No attachments provided."
        )

        email_request_json = email_request.model_dump_json(indent=2)

        # Add scheduled task context if this is a scheduled task execution
        scheduled_context = ""
        if email_request.scheduled_task_id:
            scheduled_context = self._create_scheduled_task_context(email_request.scheduled_task_id)

        base_context = f"""Email Content:
    Subject: {email_request.subject}
    From: {email_request.from_email}
    Email Date: {email_request.date}
    Recipients: {recipients}
    CC: {email_request.cc or "N/A"}
    BCC: {email_request.bcc or "N/A"}
    Body: {email_request.textContent or email_request.htmlContent or ""}

    {attachments_info}

Raw Email Request Data (for tool use):
{email_request_json}"""

        if scheduled_context:
            return f"""{scheduled_context}

{base_context}"""
        return base_context

    def _create_scheduled_task_context(self, scheduled_task_id: str) -> str:
        """
        Create context information for a scheduled task execution.

        Args:
            scheduled_task_id: The ID of the scheduled task being executed

        Returns:
            str: Formatted context explaining this is a scheduled task execution

        """
        try:
            with init_db_connection().get_session() as session:
                # Get the task information
                statement = select(Tasks).where(Tasks.task_id == scheduled_task_id)
                task = session.exec(statement).first()

                if not task:
                    return SCHEDULED_TASK_NOT_FOUND_TEMPLATE.format(scheduled_task_id=scheduled_task_id)

                # Parse the original email request
                try:
                    original_request = (
                        json.loads(task.email_request) if isinstance(task.email_request, str) else task.email_request
                    )
                except (json.JSONDecodeError, TypeError):
                    original_request = {}

                # Format the execution context
                return SCHEDULED_TASK_CONTEXT_TEMPLATE.format(
                    scheduled_task_id=scheduled_task_id,
                    created_at=task.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if task.created_at else "Unknown",
                    cron_expression=task.cron_expression,
                    original_subject=original_request.get("subject", "Unknown"),
                    original_from=original_request.get("from_email", original_request.get("from", "Unknown")),
                    task_status=task.status,
                )

        except Exception as e:
            logger.error(f"Error creating scheduled task context for {scheduled_task_id}: {e}")
            return SCHEDULED_TASK_ERROR_TEMPLATE.format(scheduled_task_id=scheduled_task_id)

    def _create_attachment_task(self, attachment_details: list[str]) -> str:
        """
        Return instructions for processing attachments, if any.

        Args:
            attachment_details: List of formatted attachment details

        Returns:
            str: Instructions for processing attachments

        """
        return f"Process these attachments:\n{chr(10).join(attachment_details)}" if attachment_details else ""

    def _create_task(self, email_request: EmailRequest, email_instructions: ProcessingInstructions) -> str:
        """
        Create a task description for the agent based on email handle instructions.

        Args:
            email_request: EmailRequest instance containing email data
            email_instructions: EmailHandleInstructions object containing processing configuration

        Returns:
            str: The task description for the agent

        """
        # process attachments if specified
        attachments = (
            self._format_attachments(email_request.attachments)
            if email_instructions.process_attachments and email_request.attachments
            else []
        )

        output_template = email_instructions.output_template

        return self._create_task_template(
            handle=email_instructions.handle,
            email_context=self._create_email_context(email_request, attachments),
            handle_specific_template=email_instructions.task_template,
            attachment_task=self._create_attachment_task(attachments),
            deep_research_mandatory=email_instructions.deep_research_mandatory,
            output_template=output_template,
            distilled_processing_instructions=email_request.distilled_processing_instructions,
        )

    def _format_attachments(self, attachments: list[EmailAttachment]) -> list[str]:
        """
        Format attachment details for inclusion in the task.

        Args:
            attachments: List of EmailAttachment objects

        Returns:
            List[str]: Formatted attachment details

        """
        return [
            f'- {att.filename} (Type: {att.contentType}, Size: {att.size} bytes)\n  EXACT FILE PATH: "{att.path}"'
            for att in attachments
        ]

    def _create_task_template(
        self,
        handle: str,
        email_context: str,
        handle_specific_template: str = "",
        attachment_task: str = "",
        deep_research_mandatory: bool = False,
        output_template: str = "",
        distilled_processing_instructions: Optional[str] = None,
    ) -> str:
        """
        Combine all task components into the final task description.

        Args:
            handle: The email handle being processed.
            email_context: The context information extracted from the email.
            handle_specific_template: Any specific template for the handle.
            attachment_task: Instructions for processing attachments.
            deep_research_mandatory: Flag indicating if deep research is mandatory.
            output_template: The output template to use.
            distilled_processing_instructions: Specific processing instructions for scheduled tasks.

        Returns:
            str: The complete task description for the agent.

        """
        # Create distilled processing instructions section for scheduled tasks
        distilled_section = (
            SCHEDULED_TASK_DISTILLED_INSTRUCTIONS_TEMPLATE.format(
                distilled_processing_instructions=distilled_processing_instructions
            )
            if distilled_processing_instructions
            else ""
        )

        # Merge the task components into a single string by listing the sections
        sections = [
            f"Process this email according to the '{handle}' instruction type.\n",
            email_context,
            distilled_section,
            RESEARCH_GUIDELINES["mandatory"] if deep_research_mandatory else RESEARCH_GUIDELINES["optional"],
            attachment_task,
            handle_specific_template,
            output_template,
            RESPONSE_GUIDELINES,
            MARKDOWN_STYLE_GUIDE,
            SECURITY_GUIDELINES,
        ]

        return "\n\n".join(filter(None, sections))

    def _process_agent_result(
        self, final_answer_obj: Any, agent_steps: list, current_email_handle: str
    ) -> DetailedEmailProcessingResult:
        processed_at_time = datetime.now().isoformat()

        # Initialize schema components
        errors_list: list[ProcessingError] = []
        email_sent_status = EmailSentStatus(status="pending", timestamp=processed_at_time)

        attachment_proc_summary: Union[str, None] = None
        processed_attachment_details: list[ProcessedAttachmentDetail] = []

        calendar_result_data: Union[CalendarResult, None] = None

        research_output_findings: Union[str, None] = None
        research_output_metadata: Union[AgentResearchMetadata, None] = None

        pdf_export_result: Union[PDFExportResult, None] = None

        final_answer_from_llm: Union[str, None] = None
        email_text_content: Union[str, None] = None
        email_html_content: Union[str, None] = None

        try:
            logger.debug(f"Processing final answer object type: {type(final_answer_obj)}")
            logger.debug(f"Processing {len(agent_steps)} agent step entries.")

            for i, step in enumerate(agent_steps):
                logger.debug(f"[Memory Step {i + 1}] Type: {type(step)}")

                tool_name = None
                tool_output = None

                if hasattr(step, "tool_calls") and isinstance(step.tool_calls, list) and len(step.tool_calls) > 0:
                    first_tool_call = step.tool_calls[0]
                    tool_name = getattr(first_tool_call, "name", None)
                    if not tool_name:
                        logger.warning(f"[Memory Step {i + 1}] Could not extract tool name from first call.")
                        tool_name = None

                    action_out = getattr(step, "action_output", None)
                    obs_out = getattr(step, "observations", None)
                    tool_output = action_out if action_out is not None else obs_out

                if tool_name and tool_output is not None:
                    needs_parsing = tool_name in [
                        "meeting_creator",
                        "attachment_processor",
                        "deep_research",
                        "pdf_export",
                        "scheduled_tasks",
                    ]
                    if isinstance(tool_output, str) and needs_parsing:
                        try:
                            tool_output = ast.literal_eval(tool_output)
                        except (ValueError, SyntaxError) as e:
                            logger.error(
                                f"[Memory Step {i + 1}] Failed to parse '{tool_name}' output: {e!s}. Content: {tool_output[:200]}..."
                            )
                            errors_list.append(
                                ProcessingError(message=f"Failed to parse {tool_name} output", details=str(e))
                            )
                            continue
                        except Exception as e:
                            logger.error(
                                f"[Memory Step {i + 1}] Unexpected error parsing '{tool_name}' output: {e!s}. Content: {tool_output[:200]}..."
                            )
                            errors_list.append(
                                ProcessingError(message=f"Unexpected error parsing {tool_name} output", details=str(e))
                            )
                            continue

                    logger.debug(
                        f"[Memory Step {i + 1}] Processing tool call: '{tool_name}', Output Type: '{type(tool_output)}'"
                    )

                    if tool_name == "attachment_processor" and isinstance(tool_output, dict):
                        attachment_proc_summary = tool_output.get("summary")
                        for attachment_data in tool_output.get("attachments", []):
                            pa_detail = ProcessedAttachmentDetail(
                                filename=attachment_data.get("filename", "unknown.file"),
                                size=attachment_data.get("size", 0),
                                type=attachment_data.get("type", "unknown"),
                            )
                            if "error" in attachment_data:
                                pa_detail.error = attachment_data["error"]
                                errors_list.append(
                                    ProcessingError(
                                        message=f"Error processing attachment {pa_detail.filename}",
                                        details=pa_detail.error,
                                    )
                                )
                            if "content" in attachment_data and isinstance(attachment_data["content"], dict):
                                if attachment_data["content"].get("caption"):
                                    pa_detail.caption = attachment_data["content"]["caption"]
                            processed_attachment_details.append(pa_detail)

                    elif tool_name == "deep_research" and isinstance(tool_output, dict):
                        research_output_findings = tool_output.get("findings")
                        research_output_metadata = AgentResearchMetadata(
                            query=tool_output.get("query"),
                            annotations=tool_output.get("annotations", []),
                            visited_urls=tool_output.get("visited_urls", []),
                            read_urls=tool_output.get("read_urls", []),
                            timestamp=tool_output.get("timestamp"),
                            usage=tool_output.get("usage", {}),
                            num_urls=tool_output.get("num_urls", 0),
                        )
                        if not research_output_findings:
                            errors_list.append(ProcessingError(message="Deep research tool returned empty findings."))

                    elif tool_name == "meeting_creator" and isinstance(tool_output, dict):
                        if tool_output.get("status") == "success" and tool_output.get("ics_content"):
                            calendar_result_data = CalendarResult(ics_content=tool_output["ics_content"])
                        else:
                            error_msg = tool_output.get("message", "Schedule generator failed or missing ICS content.")
                            errors_list.append(ProcessingError(message="Schedule Tool Error", details=error_msg))

                    elif tool_name == "pdf_export" and isinstance(tool_output, dict):
                        if tool_output.get("success"):
                            pdf_export_result = PDFExportResult(
                                filename=tool_output.get("filename", "document.pdf"),
                                file_path=tool_output.get("file_path", ""),
                                file_size=tool_output.get("file_size", 0),
                                title=tool_output.get("title", "Document"),
                                pages_estimated=tool_output.get("pages_estimated", 1),
                                mimetype=tool_output.get("mimetype", "application/pdf"),
                                temp_dir=tool_output.get("temp_dir"),
                            )
                            logger.info(f"PDF export successful: {pdf_export_result.filename}")
                        else:
                            error_msg = tool_output.get("error", "PDF export failed")
                            details = tool_output.get("details", "")
                            errors_list.append(
                                ProcessingError(message="PDF Export Error", details=f"{error_msg}. {details}")
                            )
                            logger.error(f"PDF export failed: {error_msg}")

                    elif tool_name == "scheduled_tasks" and isinstance(tool_output, dict):
                        if tool_output.get("success") and tool_output.get("task_id"):
                            logger.info(f"Scheduled task created successfully with ID: {tool_output['task_id']}")
                        else:
                            error_msg = tool_output.get("message", "Scheduled task creation failed")
                            error_type = "Scheduled Task Limit Exceeded" if tool_output.get("error") == "Task limit exceeded" else "Scheduled Task Error"
                            errors_list.append(ProcessingError(message=error_type, details=error_msg))
                            if tool_output.get("error") == "Task limit exceeded":
                                logger.warning(f"Scheduled task limit exceeded: {error_msg}")
                            else:
                                logger.error(f"Scheduled task creation failed: {error_msg}")

                    else:
                        logger.debug(
                            f"[Memory Step {i + 1}] Tool '{tool_name}' output processed (no specific handler). Output: {str(tool_output)[:200]}..."
                        )
                else:
                    logger.debug(
                        f"[Memory Step {i + 1}] Skipping step (Type: {type(step)}), not a relevant ActionStep or missing output."
                    )

            # Extract final answer from LLM
            if hasattr(final_answer_obj, "text"):
                final_answer_from_llm = str(final_answer_obj.text).strip()
                logger.debug("Extracted final answer from AgentResponse.text")
            elif isinstance(final_answer_obj, str):
                final_answer_from_llm = final_answer_obj.strip()
                logger.debug("Extracted final answer from string")
            elif hasattr(final_answer_obj, "_value"):  # Check for older AgentText structure
                final_answer_from_llm = str(final_answer_obj._value).strip()
                logger.debug("Extracted final answer from AgentText._value")
            elif hasattr(final_answer_obj, "answer"):  # Handle final_answer tool call argument
                # Check if the argument itself is the content string
                if isinstance(getattr(final_answer_obj, "answer", None), str):
                    final_answer_from_llm = str(final_answer_obj.answer).strip()
                    logger.debug("Extracted final answer from final_answer tool argument string")
                # Or if it's nested in arguments (less likely for final_answer but check)
                elif (
                    isinstance(getattr(final_answer_obj, "arguments", None), dict)
                    and "answer" in final_answer_obj.arguments
                ):
                    final_answer_from_llm = str(final_answer_obj.arguments["answer"]).strip()
                    logger.debug("Extracted final answer from final_answer tool arguments dict")
                else:
                    final_answer_from_llm = str(final_answer_obj).strip()
                    logger.warning(
                        f"Could not find specific answer attribute in final_answer object, using str(). Result: {final_answer_from_llm[:100]}..."
                    )
            else:
                final_answer_from_llm = str(final_answer_obj).strip()
                logger.warning(
                    f"Could not find specific answer attribute in final_answer object, using str(). Result: {final_answer_from_llm[:100]}..."
                )

            # Determine email body content
            email_body_content_source = research_output_findings if research_output_findings else final_answer_from_llm

            if email_body_content_source:
                signature_markers = [
                    "Best regards,\nMXtoAI Assistant",
                    "Best regards,",
                    "Warm regards,",
                    "_Feel free to reply to this email to continue our conversation._",
                    "MXtoAI Assistant",
                    "> **Disclaimer:**",
                ]
                temp_content = email_body_content_source
                for marker in signature_markers:
                    temp_content = re.sub(
                        r"^[\s\n]*" + re.escape(marker) + r".*$", "", temp_content, flags=re.IGNORECASE | re.MULTILINE
                    ).strip()

                email_text_content = self.report_formatter.format_report(
                    temp_content, format_type="text", include_signature=True
                )
                email_html_content = self.report_formatter.format_report(
                    temp_content, format_type="html", include_signature=True
                )
            else:
                fallback_msg = "I apologize, but I encountered an issue generating the detailed response. Please try again later or contact support if this issue persists."
                email_text_content = self.report_formatter.format_report(
                    fallback_msg, format_type="text", include_signature=True
                )
                email_html_content = self.report_formatter.format_report(
                    fallback_msg, format_type="html", include_signature=True
                )
                errors_list.append(ProcessingError(message="No final answer text was generated or extracted"))
                email_sent_status.status = "error"
                email_sent_status.error = "No reply text was generated"

            # Construct the final Pydantic model INSIDE the try block
            return DetailedEmailProcessingResult(
                metadata=ProcessingMetadata(
                    processed_at=processed_at_time,
                    mode=current_email_handle,  # Use the passed handle for mode
                    errors=errors_list,
                    email_sent=email_sent_status,
                ),
                email_content=EmailContentDetails(
                    text=email_text_content,
                    html=email_html_content,
                    # Assuming enhanced content is same as base for now
                    enhanced={"text": email_text_content, "html": email_html_content},
                ),
                attachments=AttachmentsProcessingResult(
                    summary=attachment_proc_summary, processed=processed_attachment_details
                ),
                calendar_data=calendar_result_data,
                research=AgentResearchOutput(
                    findings_content=research_output_findings, metadata=research_output_metadata
                )
                if research_output_findings or research_output_metadata
                else None,
                pdf_export=pdf_export_result,
            )

        except Exception as e:
            logger.exception(f"Critical error in _process_agent_result: {e!s}")
            # Ensure errors_list and email_sent_status are updated
            # If these were initialized outside and before this try-except, they might already exist.
            # Re-initialize or ensure they are correctly formed for the error state.
            # This part already handles populating errors_list and setting email_sent_status.

            # Ensure basic structure for fallback if critical error happened early
            if not errors_list:  # If the error happened before any specific error was added
                errors_list.append(ProcessingError(message="Critical error in _process_agent_result", details=str(e)))

            current_timestamp = datetime.now().isoformat()  # Use a fresh timestamp
            if email_sent_status.status != "error":  # If not already set to error by prior logic
                email_sent_status.status = "error"
                email_sent_status.error = f"Critical error in _process_agent_result: {e!s}"
                email_sent_status.timestamp = current_timestamp

            # Fallback email content if not already set
            fb_text = "I encountered a critical error processing your request during result generation."
            final_email_text = (
                email_text_content
                if email_text_content
                else self.report_formatter.format_report(fb_text, format_type="text", include_signature=True)
            )
            final_email_html = (
                email_html_content
                if email_html_content
                else self.report_formatter.format_report(fb_text, format_type="html", include_signature=True)
            )

            # Construct and return an error-state DetailedEmailProcessingResult
            return DetailedEmailProcessingResult(
                metadata=ProcessingMetadata(
                    processed_at=processed_at_time,  # or current_timestamp, consider consistency
                    mode=current_email_handle,
                    errors=errors_list,
                    email_sent=email_sent_status,
                ),
                email_content=EmailContentDetails(
                    text=final_email_text,
                    html=final_email_html,
                    enhanced={"text": final_email_text, "html": final_email_html},  # ensure enhanced also has fallback
                ),
                attachments=AttachmentsProcessingResult(
                    summary=attachment_proc_summary
                    if attachment_proc_summary
                    else None,  # Keep any partial data if available
                    processed=processed_attachment_details if processed_attachment_details else [],
                ),
                calendar_data=calendar_result_data,  # Keep any partial data
                research=AgentResearchOutput(  # Keep any partial data
                    findings_content=research_output_findings, metadata=research_output_metadata
                )
                if research_output_findings or research_output_metadata
                else None,
                pdf_export=pdf_export_result,
            )

    def process_email(
        self,
        email_request: EmailRequest,
        email_instructions: ProcessingInstructions,
    ) -> DetailedEmailProcessingResult:  # Updated return type annotation
        """
        Process an email using the agent with MCP tools loaded per-request.

        Args:
            email_request: EmailRequest instance containing email data
            email_instructions: ProcessingInstructions object containing processing configuration

        Returns:
            DetailedEmailProcessingResult: Pydantic model with structured processing results.

        """
        try:
            self.routed_model.current_handle = email_instructions

            # Get MCP tools for this request (fresh connections)
            mcp_tools = self._get_mcp_tools_for_request()

            # Create a combined tool list for this request
            all_tools = self.available_tools.copy()
            all_tools.extend(mcp_tools)

            # Create a fresh agent instance with MCP tools for this request
            request_agent = ToolCallingAgent(
                model=self.routed_model,
                tools=all_tools,
                max_steps=12,
                verbosity_level=2,
                planning_interval=4,
                name="mxtoai_email_processing_agent",
                description="I'm MXtoAI agent - an intelligent email processing agent that automates email-driven tasks and workflows. I can analyze emails, generate professional summaries and replies, conduct comprehensive research using web search and external APIs, process attachments (documents, images, PDFs), extract and create calendar events, export content to PDF, execute code for data analysis, and interact with GitHub repositories and other external services through MCP tools. I maintain professional communication standards while providing accurate, well-researched responses tailored to your specific email handling requirements.",
                provide_run_summary=True,
            )

            # Set up console logging for this agent instance
            smolagents_console = get_smolagents_console()
            if hasattr(request_agent, "logger") and hasattr(request_agent.logger, "console"):
                request_agent.logger.console = smolagents_console
            if (
                hasattr(request_agent, "monitor")
                and hasattr(request_agent.monitor, "logger")
                and hasattr(request_agent.monitor.logger, "console")
            ):
                request_agent.monitor.logger.console = smolagents_console

            task = self._create_task(email_request, email_instructions)

            logger.info("Starting agent execution...")
            final_answer_obj = self.agent.run(task, additional_args={"email_request": email_request})
            logger.info("Agent execution completed.")

            agent_steps = list(request_agent.memory.steps)
            logger.info(f"Captured {len(agent_steps)} steps from agent memory.")

            processed_result = self._process_agent_result(final_answer_obj, agent_steps, email_instructions.handle)

            if not processed_result.email_content or not processed_result.email_content.text:
                msg = "No reply text was generated by _process_agent_result"
                logger.error(msg)
                processed_result.metadata.errors.append(ProcessingError(message=msg))
                processed_result.metadata.email_sent.status = "error"
                processed_result.metadata.email_sent.error = msg

                logger.info(f"Email processed (but no reply text generated) with handle: {email_instructions.handle}")
                return processed_result

            logger.info(f"Email processed successfully with handle: {email_instructions.handle}")
            return processed_result

        except Exception as e:
            error_msg = f"Critical error in email processing: {e!s}"
            logger.error(error_msg, exc_info=True)

            # Construct a DetailedEmailProcessingResult for error cases
            now_iso = datetime.now().isoformat()
            return DetailedEmailProcessingResult(
                metadata=ProcessingMetadata(
                    processed_at=now_iso,
                    mode=email_instructions.handle if email_instructions else "unknown",
                    errors=[ProcessingError(message=error_msg, details=str(e))],
                    email_sent=EmailSentStatus(status="error", error=error_msg, timestamp=now_iso),
                ),
                email_content=EmailContentDetails(
                    text=self.report_formatter.format_report(
                        "I encountered a critical error processing your request.",
                        format_type="text",
                        include_signature=True,
                    ),
                    html=self.report_formatter.format_report(
                        "I encountered a critical error processing your request.",
                        format_type="html",
                        include_signature=True,
                    ),
                    enhanced={"text": None, "html": None},
                ),
                attachments=AttachmentsProcessingResult(processed=[]),
                calendar_data=None,
                research=None,
                pdf_export=None,
            )

    def _create_limited_scheduled_tasks_tool(self, email_request: EmailRequest) -> Tool:
        """
        Create a scheduled tasks tool with a call limit wrapper.

        Args:
            email_request: The email request data

        Returns:
            Tool: Wrapped scheduled tasks tool with call limiting

        """
        # Create the base tool
        base_tool = ScheduledTasksTool(email_request=email_request)

        # Create a counter to track calls
        call_count = {"count": 0}
        max_calls = SCHEDULED_TASKS_MAX_PER_EMAIL

        # Store original forward method
        original_forward = base_tool.forward

        def limited_forward(*args, **kwargs):
            """Wrapper that limits scheduled task calls to 5 per email."""
            if call_count["count"] >= max_calls:
                logger.warning(f"Scheduled task limit reached ({max_calls} tasks per email). Rejecting additional task creation.")
                return {
                    "success": False,
                    "error": "Task limit exceeded",
                    "message": f"Maximum of {max_calls} scheduled tasks allowed per email. This limit helps prevent excessive automation.",
                    "tasks_created": call_count["count"],
                    "max_allowed": max_calls,
                }

            # Increment counter before calling
            call_count["count"] += 1
            logger.info(f"Creating scheduled task {call_count['count']}/{max_calls}")

            # Call the original method
            result = original_forward(*args, **kwargs)

            # If the call failed, decrement the counter
            if not result.get("success", False):
                call_count["count"] -= 1
                logger.info(f"Scheduled task creation failed, decremented counter to {call_count['count']}")

            return result

        # Replace the forward method
        base_tool.forward = limited_forward

        return base_tool

    def _get_mcp_tools_for_request(self) -> list[Tool]:
        """
        Get MCP tools for a single request using Smolagents native MCP support.
        This avoids subprocess hanging issues in dramatiq workers.
        """
        import os
        import threading
        import toml
        from pathlib import Path

        mcp_tools = []

        # Check if MCP is disabled via environment variable
        if os.getenv("MXTOAI_DISABLE_MCP_IN_WORKERS", "false").lower() == "true":
            logger.info("MCP disabled via MXTOAI_DISABLE_MCP_IN_WORKERS environment variable")
            return mcp_tools

        # Check if we're in a multiprocessing context (dramatiq worker)
        try:
            import multiprocessing
            process_name = multiprocessing.current_process().name
            is_dramatiq_worker = "dramatiq" in process_name.lower() or "worker" in process_name.lower() or "process-" in process_name.lower()

            # Check environment variables that dramatiq might set
            dramatiq_env_vars = [
                'DRAMATIQ_PROCESSES', 'DRAMATIQ_THREADS', 'DRAMATIQ_BROKER_URL',
                'PROMETHEUS_MULTIPROC_DIR', 'prometheus_multiproc_dir'
            ]
            dramatiq_env_found = [var for var in dramatiq_env_vars if var in os.environ]
            if dramatiq_env_found:
                is_dramatiq_worker = True

            # IMMEDIATE FIX: Disable MCP in dramatiq workers to prevent hanging
            if is_dramatiq_worker:
                logger.warning("🚫 MCP disabled in dramatiq worker environment to prevent subprocess hanging")
                logger.warning("🔧 Agent will continue with non-MCP tools only")
                return mcp_tools

        except Exception as e:
            logger.info(f"🔍 DEBUG: Multiprocessing check failed: {e}")

        try:
            from smolagents.mcp_client import MCPClient
            from mcp import StdioServerParameters

            # Debug: Log environment information
            logger.info(f"🔍 DEBUG: Starting MCP tools loading...")
            logger.info(f"🔍 DEBUG: Process ID: {os.getpid()}")
            logger.info(f"🔍 DEBUG: Parent Process ID: {os.getppid()}")
            logger.info(f"🔍 DEBUG: Thread ID: {threading.get_ident()}")
            logger.info(f"🔍 DEBUG: Active thread count: {threading.active_count()}")

            # Load MCP config from mcp.toml
            config_path = Path("mcp.toml")
            if not config_path.exists():
                logger.info("No mcp.toml file found, skipping MCP tools")
                return mcp_tools

            try:
                logger.info("🔍 DEBUG: Reading mcp.toml config...")
                with open(config_path, "r") as f:
                    config_data = toml.load(f)

                mcp_servers = config_data.get("mcp_servers", {})
                enabled_servers = {name: config for name, config in mcp_servers.items() if config.get("enabled", False)}

                if not enabled_servers:
                    logger.info("No enabled MCP servers found in mcp.toml")
                    return mcp_tools

                logger.info(f"Loading MCP tools from {len(enabled_servers)} enabled servers...")
                logger.info(f"🔍 DEBUG: Enabled servers: {list(enabled_servers.keys())}")

                # Use Smolagents native MCP client for each server
                for server_name, server_config in enabled_servers.items():
                    logger.info(f"🔍 DEBUG: Processing server: {server_name}")
                    try:
                        server_type = server_config.get("type", "stdio")
                        logger.info(f"🔍 DEBUG: Server type: {server_type}")

                        if server_type == "stdio":
                            command = server_config.get("command")
                            args = server_config.get("args", [])
                            env = server_config.get("env", {})

                            logger.info(f"🔍 DEBUG: Command: {command}")
                            logger.info(f"🔍 DEBUG: Args: {args}")
                            logger.info(f"🔍 DEBUG: Env vars: {list(env.keys())}")

                            if not command:
                                logger.warning(f"No command specified for stdio server {server_name}")
                                continue

                            # Merge environment variables with os.environ
                            merged_env = {**os.environ, **env}
                            logger.info(f"🔍 DEBUG: Merged environment has {len(merged_env)} variables")

                            # Test if the command exists and is accessible
                            logger.info("🔍 DEBUG: Testing command accessibility...")
                            try:
                                import subprocess
                                result = subprocess.run([command, "--version"],
                                                      capture_output=True,
                                                      text=True,
                                                      timeout=10,
                                                      env=merged_env)
                                logger.info(f"🔍 DEBUG: Command test result: {result.returncode}")
                                if result.stdout:
                                    logger.info(f"🔍 DEBUG: Command stdout: {result.stdout[:100]}")
                                if result.stderr:
                                    logger.info(f"🔍 DEBUG: Command stderr: {result.stderr[:100]}")
                            except Exception as cmd_e:
                                logger.warning(f"🔍 DEBUG: Command test failed: {cmd_e}")

                            stdio_params = StdioServerParameters(
                                command=command,
                                args=args,
                                env=merged_env,
                            )
                            logger.info(f"🔍 DEBUG: Created StdioServerParameters: {stdio_params}")

                            # Use Smolagents native MCP client with timeout
                            logger.info(f"🔍 DEBUG: Using Smolagents native MCP client for {server_name}...")

                            try:
                                # Use MCPClient directly (it uses MCPAdapt internally but may handle errors better)
                                with MCPClient(stdio_params) as server_tools:
                                    logger.info(f"🔍 DEBUG: Successfully loaded {len(server_tools)} tools from {server_name}")
                                    mcp_tools.extend(server_tools)

                            except Exception as mcp_e:
                                logger.error(f"🔍 DEBUG: Error loading MCP server {server_name}: {mcp_e}", exc_info=True)
                                logger.warning(f"MCP server {server_name} failed to load - skipping")
                                continue

                        elif server_type in ["sse", "streamable-http"]:
                            logger.info(f"🔍 DEBUG: Processing HTTP server type: {server_type}")
                            # HTTP-based servers should work better in worker environments
                            url = server_config.get("url")
                            if not url:
                                logger.warning(f"No URL specified for {server_type} server {server_name}")
                                continue

                            http_params = {
                                "url": url,
                                "transport": server_type
                            }
                            logger.info(f"🔍 DEBUG: Created HTTP parameters: {http_params}")

                            try:
                                with MCPClient(http_params) as server_tools:
                                    logger.info(f"🔍 DEBUG: Successfully loaded {len(server_tools)} tools from {server_name}")
                                    mcp_tools.extend(server_tools)
                            except Exception as http_e:
                                logger.error(f"🔍 DEBUG: Error loading HTTP MCP server {server_name}: {http_e}", exc_info=True)
                                continue

                        else:
                            logger.warning(f"Unsupported MCP server type: {server_type} for {server_name}")

                    except Exception as e:
                        logger.error(f"🔍 DEBUG: Failed to load MCP server {server_name}: {e}", exc_info=True)
                        continue

            except Exception as e:
                logger.error(f"Failed to parse mcp.toml: {e}")

        except ImportError as e:
            logger.warning(f"MCP dependencies not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}", exc_info=True)

        if mcp_tools:
            logger.info(f"Successfully loaded {len(mcp_tools)} MCP tools total")
        else:
            logger.info("No MCP tools loaded")

        return mcp_tools
