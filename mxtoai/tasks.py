import asyncio
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import dramatiq
from dotenv import load_dotenv
from dramatiq.brokers.rabbitmq import RabbitmqBroker

from mxtoai import exceptions
from mxtoai._logging import get_logger
from mxtoai.agents.email_agent import EmailAgent
from mxtoai.config import SKIP_EMAIL_DELIVERY
from mxtoai.dependencies import processing_instructions_resolver
from mxtoai.email_sender import EmailSender
from mxtoai.schemas import (
    AttachmentsProcessingResult,
    DetailedEmailProcessingResult,
    EmailContentDetails,
    EmailRequest,
    EmailSentStatus,
    ProcessingError,
    ProcessingInstructions,
    ProcessingMetadata,
)
from mxtoai.validators import check_task_idempotency, mark_email_as_processed

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Build RabbitMQ URL from environment variables (Broker)
# Include heartbeat as a query parameter in the URL
RABBITMQ_HEARTBEAT = os.getenv("RABBITMQ_HEARTBEAT", "5")
RABBITMQ_URL = f"amqp://{os.getenv('RABBITMQ_USER', 'guest')}:{os.getenv('RABBITMQ_PASSWORD', 'guest')}@{os.getenv('RABBITMQ_HOST', 'localhost')}:{os.getenv('RABBITMQ_PORT', '5672')}{os.getenv('RABBITMQ_VHOST', '/')}?heartbeat={RABBITMQ_HEARTBEAT}"

# Initialize RabbitMQ broker
rabbitmq_broker = RabbitmqBroker(
    url=RABBITMQ_URL,
    confirm_delivery=True,  # Ensures messages are delivered
)
dramatiq.set_broker(rabbitmq_broker)


def cleanup_attachments(email_attachments_dir: str) -> None:
    """
    Clean up attachments after processing.

    Args:
        email_attachments_dir: Directory containing email attachments

    """
    try:
        dir_path = Path(email_attachments_dir)
        if dir_path.exists():
            for file in dir_path.iterdir():
                try:
                    file.unlink()
                    logger.debug(f"Deleted attachment: {file}")
                except Exception as e:
                    logger.error(f"Failed to delete file {file}: {e!s}")
            dir_path.rmdir()
            logger.info(f"Cleaned up attachments directory: {email_attachments_dir}")
    except Exception as e:
        logger.exception(f"Error cleaning up attachments: {e!s}")


def should_retry(retries_so_far: int, exception: Exception) -> bool:
    """
    Determine whether to retry the task based on the exception and retry count.

    Args:
        retries_so_far: Number of retries attempted
        exception: Exception raised during task execution

    Returns:
        bool: True if the task should be retried, False otherwise

    """
    logger.warning(f"Retrying task after exception: {exception!s}, retries so far: {retries_so_far}")
    return retries_so_far < 3


@dramatiq.actor(retry_when=should_retry, min_backoff=60 * 1000, time_limit=600000)
def process_email_task(
    email_data: dict[str, Any],
    email_attachments_dir: str,
    attachment_info: list[dict[str, Any]],
    scheduled_task_id: Optional[str] = None
) -> DetailedEmailProcessingResult:
    """
    Dramatiq task for processing emails asynchronously.

    Args:
        email_data: Dictionary containing email request data
        email_attachments_dir: Directory containing email attachments
        attachment_info: List of attachment information dictionaries
        scheduled_task_id: Optional task ID if this is a scheduled task

    Returns:
        DetailedEmailProcessingResult: The result of the email processing.

    """
    email_request = EmailRequest(**email_data)

        # Check for duplicate processing using Redis (idempotency check)
    message_id = email_request.messageId

    if check_task_idempotency(message_id):
        # Return a minimal result indicating it was already processed
        now_iso = datetime.now().isoformat()
        return DetailedEmailProcessingResult(
            metadata=ProcessingMetadata(
                processed_at=now_iso,
                mode="duplicate",
                errors=[ProcessingError(message="Email already processed (duplicate)")],
                email_sent=EmailSentStatus(
                    status="duplicate",
                    message_id="duplicate",
                    timestamp=now_iso,
                ),
            ),
            email_content=EmailContentDetails(text=None, html=None, enhanced=None),
            attachments=AttachmentsProcessingResult(processed=[]),
            calendar_data=None,
            research=None,
            pdf_export=None,
        )

    # For scheduled tasks, use distilled_alias if available, otherwise fall back to email handle
    if scheduled_task_id and email_request.distilled_alias:
        handle = email_request.distilled_alias.value
        logger.info(f"Processing scheduled task {scheduled_task_id} using distilled alias: {handle}")
    else:
        handle = email_request.to.split("@")[0].lower()
        if scheduled_task_id:
            logger.info(f"Processing scheduled task {scheduled_task_id} for handle: {handle}")
        else:
            logger.info(f"Processing regular email for handle: {handle}")

    now_iso = datetime.now().isoformat()  # Define now_iso earlier for use in error cases

    try:
        email_instructions = processing_instructions_resolver(handle)
    except exceptions.UnspportedHandleException as e:  # Catch specific exception
        logger.error(f"Unsupported email handle: {handle}. Error: {e!s}")
        return DetailedEmailProcessingResult(
            metadata=ProcessingMetadata(
                processed_at=now_iso,
                mode=handle,
                errors=[ProcessingError(message=f"Unsupported email handle: {handle}", details=str(e))],
                email_sent=EmailSentStatus(
                    status="error", error=f"Unsupported email handle: {handle} - {e!s}", timestamp=now_iso
                ),
            ),
            email_content=EmailContentDetails(text=None, html=None, enhanced=None),
            attachments=AttachmentsProcessingResult(processed=[]),
            calendar_data=None,
            research=None,
            pdf_export=None,
        )

    email_agent = EmailAgent(email_request=email_request)

    if email_instructions.deep_research_mandatory and email_agent.research_tool:
        email_agent.research_tool.enable_deep_research()
    elif email_agent.research_tool:  # Ensure research_tool exists before trying to disable
        email_agent.research_tool.disable_deep_research()

    if email_request.attachments and attachment_info:
        valid_attachments = []
        for attachment_model, info_dict in zip(email_request.attachments, attachment_info, strict=False):
            try:
                if not Path(info_dict["path"]).exists():
                    logger.error(f"Attachment file not found: {info_dict['path']}")
                    continue
                attachment_model.path = info_dict["path"]
                attachment_model.contentType = (
                    info_dict.get("type") or info_dict.get("contentType") or "application/octet-stream"
                )
                attachment_model.size = info_dict.get("size", 0)
                valid_attachments.append(attachment_model)
            except Exception as e:
                logger.error(f"Error processing attachment {attachment_model.filename}: {e!s}")
        email_request.attachments = valid_attachments

    # Set scheduled task ID in email request for the agent to access
    if scheduled_task_id:
        # Add the task ID to the email data for the agent to process
        email_request.scheduled_task_id = scheduled_task_id

    processing_result = email_agent.process_email(email_request, email_instructions)

    # Add task ID to email content if this is a scheduled task
    if scheduled_task_id and processing_result.email_content:
        # Append task ID to both text and HTML content
        task_id_note = f"\n\n---\nTask ID: {scheduled_task_id}"
        task_id_note_html = f"<br/><br/><hr/><p><strong>Task ID:</strong> {scheduled_task_id}</p>"

        if processing_result.email_content.text:
            processing_result.email_content.text += task_id_note

        if processing_result.email_content.html:
            # Insert before closing body tag if present, otherwise append
            if "</body>" in processing_result.email_content.html:
                processing_result.email_content.html = processing_result.email_content.html.replace(
                    "</body>", f"{task_id_note_html}</body>"
                )
            else:
                processing_result.email_content.html += task_id_note_html

    if processing_result.email_content and processing_result.email_content.text:
        if email_request.from_email in SKIP_EMAIL_DELIVERY:
            logger.info(f"Skipping email delivery for test email: {email_request.from_email}")
            processing_result.metadata.email_sent.status = "skipped"
            processing_result.metadata.email_sent.message_id = "skipped"

            # Mark as processed in Redis even for skipped emails
            mark_email_as_processed(message_id)
        else:
            attachments_to_send = []
            if processing_result.calendar_data and processing_result.calendar_data.ics_content:
                attachments_to_send.append(
                    {
                        "filename": "invite.ics",
                        "content": processing_result.calendar_data.ics_content,
                        "mimetype": "text/calendar",
                    }
                )
                logger.info("Prepared invite.ics for attachment in task.")

            # Add PDF export attachment if available
            if processing_result.pdf_export and processing_result.pdf_export.file_path:
                try:
                    # Read the PDF file content
                    with open(processing_result.pdf_export.file_path, "rb") as pdf_file:
                        pdf_content = pdf_file.read()

                    attachments_to_send.append(
                        {
                            "filename": processing_result.pdf_export.filename,
                            "content": pdf_content,
                            "mimetype": processing_result.pdf_export.mimetype,
                        }
                    )
                    logger.info(f"Prepared {processing_result.pdf_export.filename} for attachment in task.")

                    # Clean up the temporary PDF file
                    os.unlink(processing_result.pdf_export.file_path)
                    logger.info(f"Cleaned up temporary PDF file: {processing_result.pdf_export.file_path}")

                    # Clean up the PDF tool's temporary directory using tracked temp_dir
                    if processing_result.pdf_export.temp_dir:
                        pdf_temp_dir = processing_result.pdf_export.temp_dir
                        if pdf_temp_dir and os.path.exists(pdf_temp_dir):
                            shutil.rmtree(pdf_temp_dir, ignore_errors=True)
                            logger.info(f"Cleaned up PDF tool temp directory: {pdf_temp_dir}")
                    else:
                        # Fallback: extract parent directory from the PDF file path
                        pdf_temp_dir = Path(processing_result.pdf_export.file_path).parent
                        if pdf_temp_dir.exists():
                            shutil.rmtree(pdf_temp_dir, ignore_errors=True)
                            logger.info(f"Cleaned up PDF tool temp directory (fallback): {pdf_temp_dir}")

                except Exception as pdf_error:
                    logger.error(f"Failed to attach PDF file: {pdf_error}")
                    # Continue without the PDF attachment rather than failing the entire email

            original_email_details = {
                "from": email_request.from_email,
                "to": email_request.to,
                "subject": email_request.subject,
                "messageId": email_request.messageId,
                "references": email_request.references,
                "cc": email_request.cc,
            }
            try:
                sender = EmailSender()
                email_sent_response = asyncio.run(
                    sender.send_reply(
                        original_email_details,
                        reply_text=processing_result.email_content.text,
                        reply_html=processing_result.email_content.html,
                        attachments=attachments_to_send,
                    )
                )
                processing_result.metadata.email_sent.status = email_sent_response.get(
                    "status", "sent"
                )  # Or map more carefully
                processing_result.metadata.email_sent.message_id = email_sent_response.get("MessageId")
                if email_sent_response.get("status") == "error":
                    processing_result.metadata.email_sent.error = email_sent_response.get("error", "Unknown send error")
                # Mark as processed in Redis after successful email sending
                else:
                    mark_email_as_processed(message_id)

            except Exception as send_err:
                logger.error(f"Error initializing EmailSender or sending reply: {send_err!s}", exc_info=True)
                processing_result.metadata.email_sent.status = "error"
                processing_result.metadata.email_sent.error = str(send_err)
                processing_result.metadata.email_sent.message_id = "error"

    # Log the processing result (consider converting Pydantic model to dict for json.dumps if needed)
    try:
        # Attempt to dump the Pydantic model directly, or convert to dict if complex logging is needed
        loggable_metadata = processing_result.metadata.model_dump(mode="json")
        logger.info(f"Email processed status: {loggable_metadata.get('email_sent', {}).get('status')}")
    except Exception as log_e:
        logger.error(f"Error serializing processing_result for logging: {log_e!s}")
        logger.info(f"Email processed. Status: {processing_result.metadata.email_sent.status}")  # Fallback basic log

    if email_attachments_dir:
        cleanup_attachments(email_attachments_dir)

    return processing_result
