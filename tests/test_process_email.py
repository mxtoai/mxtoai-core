import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import select

from mxtoai.db import init_db_connection
from mxtoai.email_handles import DEFAULT_EMAIL_HANDLES
from mxtoai.models.models import Tasks, TaskStatus
from mxtoai.schemas import (
    AttachmentsProcessingResult,
    DetailedEmailProcessingResult,
    EmailContentDetails,
    EmailSentStatus,
    ProcessingError,
    ProcessingMetadata,
)
from mxtoai.tasks import process_email_task

AttachmentFileContent = tuple[str, bytes, str]  # (filename, content_bytes, content_type)


@pytest.fixture
def prepare_email_request_data(tmp_path):
    def _prepare(
        to_email: str = "ask@mxtoai.com",
        from_email: str = "sender.test@example.com",
        subject: str = "Test Subject",
        text_content: str = "This is a test email.",
        html_content: str = "<p>This is a test email.</p>",
        message_id: str = "<test-message-id-default>",
        attachments_data: Optional[
            list[AttachmentFileContent]
        ] = None,  # List of (filename, content_bytes, content_type)
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        """
        Prepares email_data, email_attachments_dir_str, and attachment_info_list.
        `attachments_data` is a list of tuples: (filename, content_bytes, content_type)
        """
        attachments_dir_path = tmp_path / "attachments"
        attachments_dir_path.mkdir(exist_ok=True)

        email_attachments_schema_list: list[dict[str, Any]] = []
        attachment_info_list_for_task: list[dict[str, Any]] = []

        if attachments_data:
            for idx, (filename, content_bytes, content_type) in enumerate(attachments_data):
                dummy_file_path = attachments_dir_path / f"{idx}_{filename}"
                dummy_file_path.write_bytes(content_bytes)
                dummy_file_size = dummy_file_path.stat().st_size

                email_attachments_schema_list.append(
                    {"filename": filename, "contentType": content_type, "size": dummy_file_size}
                )
                attachment_info_list_for_task.append(
                    {"path": str(dummy_file_path), "filename": filename, "type": content_type, "size": dummy_file_size}
                )

        email_data = {
            "from_email": from_email,
            "to": to_email,
            "subject": subject,
            "textContent": text_content,
            "htmlContent": html_content,
            "messageId": message_id,
            "attachments": email_attachments_schema_list,
            "recipients": [to_email.split("@")[0] + "@mxtoai.com"],  # Simplified recipient
            "date": "2023-01-01T12:00:00Z",
        }

        return email_data, str(attachments_dir_path), attachment_info_list_for_task

    return _prepare


def _assert_basic_successful_processing(
    result: DetailedEmailProcessingResult,
    expected_handle: str,
    expect_reply_sent: bool = True,
    attachments_cleaned_up_dir: Optional[str] = None,
):
    """Helper function for common assertions on successful processing results."""
    assert isinstance(result, DetailedEmailProcessingResult), "Return type mismatch"
    assert result.metadata.mode == expected_handle, f"Expected handle {expected_handle}, got {result.metadata.mode}"
    assert not result.metadata.errors, f"Expected no errors, but got: {result.metadata.errors}"

    assert result.email_content is not None
    assert result.email_content.text is not None, "Reply text should not be None"
    assert len(result.email_content.text) > 0, "Reply text is empty"
    assert result.email_content.html is not None, "Reply HTML should not be None"
    assert len(result.email_content.html) > 0, "Reply HTML is empty"

    if expect_reply_sent:
        assert result.metadata.email_sent.status == "sent", "Email not marked as sent"
        assert result.metadata.email_sent.message_id == "mocked_message_id_happy_path", "Message ID mismatch"
    else:
        # Could be 'pending' or 'skipped' etc. depending on other factors not covered by this basic assert
        assert result.metadata.email_sent.status != "error", "Email marked as error when not expected"

    if attachments_cleaned_up_dir:
        assert not Path(attachments_cleaned_up_dir).exists(), (
            f"Attachments directory '{attachments_cleaned_up_dir}' was not cleaned up."
        )


# --- Existing Happy Path Test (adapted to use new fixture) ---
def test_process_email_task_happy_path_with_attachment(prepare_email_request_data):
    """
    Tests the happy path for process_email_task with an attachment.
    Only EmailSender.send_reply is mocked.
    """
    attachment_content = ("test_attachment.txt", b"This is a test attachment.", "text/plain")
    email_data, email_attachments_dir_str, attachment_info = prepare_email_request_data(
        to_email="ask@mxtoai.com", attachments_data=[attachment_content]
    )

    assert Path(email_attachments_dir_str).exists()
    assert (Path(email_attachments_dir_str) / f"0_{attachment_content[0]}").exists()  # Check specific file

    with patch("mxtoai.tasks.EmailSender") as MockEmailSender:  # noqa: N806
        mock_sender_instance = MockEmailSender.return_value

        async def mock_async_send_reply(*args, **kwargs):
            return {"MessageId": "mocked_message_id_happy_path", "status": "sent"}

        mock_sender_instance.send_reply = MagicMock(side_effect=mock_async_send_reply)

        returned_result = process_email_task.fn(
            email_data=email_data, email_attachments_dir=email_attachments_dir_str, attachment_info=attachment_info
        )

        mock_sender_instance.send_reply.assert_called_once()
        call_args = mock_sender_instance.send_reply.call_args
        original_email_details_arg = call_args[0][0]
        assert original_email_details_arg["from"] == email_data["from_email"]

        _assert_basic_successful_processing(
            returned_result, expected_handle="ask", attachments_cleaned_up_dir=email_attachments_dir_str
        )


# --- New Test for Future/Remind Handle ---
def test_process_email_task_future_remind_handle(prepare_email_request_data):
    """
    Tests the future/remind email handle that schedules tasks for later execution.
    Uses actual test database to verify task creation.
    """
    # Create an email with a specific future task request
    future_content = "Remind me to book a doctor's appointment next Monday at 10 AM."

    email_data, email_attachments_dir_str, attachment_info = prepare_email_request_data(
        to_email="remind@mxtoai.com",
        subject="Doctor Appointment Reminder",
        text_content=future_content,
    )

    with (
        patch("mxtoai.tasks.EmailSender") as MockEmailSender,  # noqa: N806
        patch("mxtoai.scheduling.scheduler.Scheduler.add_job") as mock_add_scheduled_job,
    ):
        mock_add_scheduled_job.return_value = None
        mock_sender_instance = MockEmailSender.return_value

        async def mock_async_send_reply(*args, **kwargs):
            return {"MessageId": "mocked_message_id_happy_path", "status": "sent"}

        mock_sender_instance.send_reply = MagicMock(side_effect=mock_async_send_reply)

        # Process the email
        returned_result = process_email_task.fn(
            email_data=email_data, email_attachments_dir=email_attachments_dir_str, attachment_info=attachment_info
        )

        # Verify basic processing success
        _assert_basic_successful_processing(
            returned_result,
            expected_handle="schedule",  # Remind aliases to schedule in current implementation
            attachments_cleaned_up_dir=email_attachments_dir_str if attachment_info else None,
        )

        # Check the actual test database for the created task
        db_connection = init_db_connection()
        with db_connection.get_session() as session:
            tasks = session.exec(select(Tasks)).all()
            assert len(tasks) > 0, "Expected at least one scheduled task to be created"

            # Verify the task has appropriate details
            task = tasks[0]
            assert task.task_id is not None, "Task ID should be set"
            assert task.cron_expression is not None, "Cron expression should be set"

        # Check for task confirmation in the returned email content (text and HTML versions)
        email_content_text = returned_result.email_content.text or ""
        email_content_html = returned_result.email_content.html or ""

        # Check for task confirmation in the response content
        has_task_reference = any(term in email_content_text for term in ["Task", "Scheduled", "scheduled", "task"])
        has_task_reference_html = any(term in email_content_html for term in ["Task", "Scheduled", "scheduled", "task"])

        assert has_task_reference or has_task_reference_html, (
            f"Email content doesn't mention task. Text: {email_content_text[:100]}..."
        )

        # The response should include the original request
        has_appointment_reference = any(term in email_content_text.lower() for term in ["doctor", "appointment"])
        has_appointment_reference_html = any(term in email_content_html.lower() for term in ["doctor", "appointment"])

        assert has_appointment_reference or has_appointment_reference_html, (
            "Email content doesn't include original request"
        )

        # Check if Next Occurrence or similar timing info is in the response
        timing_terms = ["next occurrence", "scheduled for", "next run", "will be processed", "at the scheduled time"]
        has_timing_info = any(term in email_content_text.lower() for term in timing_terms)
        has_timing_info_html = any(term in email_content_html.lower() for term in timing_terms)

        assert has_timing_info or has_timing_info_html, "Email content doesn't include timing information"

        # Verify the email was sent
        mock_sender_instance.send_reply.assert_called_once()


# --- New Test Cases ---
def test_process_email_task_unsupported_handle(prepare_email_request_data):
    """Tests behavior when an unsupported email handle is provided."""
    unsupported_handle = "nonexistenthandle@mxtoai.com"
    email_data, email_attachments_dir_str, attachment_info = prepare_email_request_data(to_email=unsupported_handle)

    # No mocks needed as it should fail before agent or sender
    returned_result = process_email_task.fn(
        email_data=email_data, email_attachments_dir=email_attachments_dir_str, attachment_info=attachment_info
    )

    assert isinstance(returned_result, DetailedEmailProcessingResult)
    assert returned_result.metadata.mode == unsupported_handle.split("@")[0]
    assert len(returned_result.metadata.errors) == 1
    assert "Unsupported email handle" in returned_result.metadata.errors[0].message
    assert returned_result.metadata.email_sent.status == "error"
    assert "Unsupported email handle" in returned_result.metadata.email_sent.error
    # The error detail from the exception should be in the returned result
    assert "This email handle is not supported" in returned_result.metadata.errors[0].details

    # Attachments dir might be created but should not be cleaned if processing stops early
    # or if it was never relevant. If it was created, it might still exist.
    # For this test, the main focus is the error state.


def test_process_email_task_agent_exception(prepare_email_request_data):
    """Tests behavior when EmailAgent.process_email returns a result indicating an internal error."""
    email_data, email_attachments_dir_str, attachment_info = prepare_email_request_data(to_email="ask@mxtoai.com")
    now_iso = datetime.now(timezone.utc).isoformat()  # For constructing mock error response

    # Prepare a mock error response that EmailAgent.process_email would return
    mock_agent_error_result = DetailedEmailProcessingResult(
        metadata=ProcessingMetadata(
            processed_at=now_iso,
            mode="ask",
            errors=[
                ProcessingError(
                    message="Critical error during agent processing", details="Simulated agent internal crash"
                )
            ],
            email_sent=EmailSentStatus(status="error", error="Simulated agent internal crash", timestamp=now_iso),
        ),
        email_content=EmailContentDetails(text=None, html=None, enhanced=None),
        attachments=AttachmentsProcessingResult(processed=[]),
        calendar_data=None,
        research=None,
    )

    with (
        patch(
            "mxtoai.tasks.EmailAgent.process_email", return_value=mock_agent_error_result
        ) as mock_agent_process_email,
        patch("mxtoai.tasks.EmailSender") as mock_email_sender_class,
    ):
        mock_sender_instance = mock_email_sender_class.return_value

        async def mock_async_send_reply(*args, **kwargs):
            return {"MessageId": "should_not_be_called", "status": "sent"}

        mock_sender_instance.send_reply = MagicMock(side_effect=mock_async_send_reply)

        returned_task_result = process_email_task.fn(
            email_data=email_data, email_attachments_dir=email_attachments_dir_str, attachment_info=attachment_info
        )

    assert isinstance(returned_task_result, DetailedEmailProcessingResult)
    # The result from the task should be the same as what the agent returned in its error state
    assert len(returned_task_result.metadata.errors) > 0
    assert returned_task_result.metadata.errors[0].message == "Critical error during agent processing"
    assert returned_task_result.metadata.errors[0].details == "Simulated agent internal crash"
    assert returned_task_result.metadata.email_sent.status == "error"
    assert returned_task_result.metadata.email_sent.error == "Simulated agent internal crash"

    mock_agent_process_email.assert_called_once()
    mock_sender_instance.send_reply.assert_not_called()  # Reply should not be attempted if agent indicates error

    assert not Path(email_attachments_dir_str).exists(), (
        f"Attachments directory '{email_attachments_dir_str}' was not cleaned up even after agent error."
    )


# --- Individual Handle Tests for Better Parallelization ---


def test_process_email_task_summarize_handle(prepare_email_request_data):
    """Test summarize handle specifically."""
    _test_single_handle("summarize", prepare_email_request_data)


def test_process_email_task_research_handle(prepare_email_request_data):
    """Test research handle specifically."""
    _test_single_handle("research", prepare_email_request_data)


def test_process_email_task_simplify_handle(prepare_email_request_data):
    """Test simplify handle specifically."""
    _test_single_handle("simplify", prepare_email_request_data)


def test_process_email_task_ask_handle(prepare_email_request_data):
    """Test ask handle specifically."""
    _test_single_handle("ask", prepare_email_request_data)


def test_process_email_task_fact_check_handle(prepare_email_request_data):
    """Test fact-check handle specifically."""
    _test_single_handle("fact-check", prepare_email_request_data)


def test_process_email_task_background_research_handle(prepare_email_request_data):
    """Test background-research handle specifically."""
    _test_single_handle("background-research", prepare_email_request_data)


def test_process_email_task_translate_handle(prepare_email_request_data):
    """Test translate handle specifically."""
    _test_single_handle("translate", prepare_email_request_data)


def test_process_email_task_meeting_handle(prepare_email_request_data):
    """Test meeting handle specifically."""
    _test_single_handle("meeting", prepare_email_request_data)


@pytest.mark.flaky(retries=3, delay=1)
def test_process_email_task_pdf_handle(prepare_email_request_data):
    """Test pdf handle specifically."""
    _test_single_handle("pdf", prepare_email_request_data)


def test_process_email_task_schedule_handle(prepare_email_request_data):
    """Test schedule handle specifically."""
    _test_single_handle("schedule", prepare_email_request_data)


def _test_single_handle(handle_name: str, prepare_email_request_data):
    """Helper function to test a single email handle."""
    # Find the handle instructions for the given handle name
    handle_instructions = None
    for handle in DEFAULT_EMAIL_HANDLES:
        if handle.handle == handle_name:
            handle_instructions = handle
            break

    if not handle_instructions:
        pytest.fail(f"Handle '{handle_name}' not found in DEFAULT_EMAIL_HANDLES")

    # Use the same logic as the parametrized test
    to_email_address = f"{handle_instructions.handle}@mxtoai.com"

    attachments_for_test: Optional[list[AttachmentFileContent]] = None
    is_schedule_handle = handle_instructions.handle == "schedule"
    is_meeting_handle = handle_instructions.handle == "meeting"

    # Specific setup for different handles
    subject_for_test = f"Test for {handle_instructions.handle}"
    text_content_for_test = f"This is a test email for the '{handle_instructions.handle}' handle."

    if is_schedule_handle:
        # Schedule handle should test future task scheduling, not meeting creation
        text_content_for_test = (
            "Please remind me every Monday at 9 AM to review the weekly sales report starting next week."
        )
        # Add a dummy file if process_attachments is True for schedule
        if handle_instructions.process_attachments:
            attachments_for_test = [("schedule_context.txt", b"Weekly report context.", "text/plain")]
    elif is_meeting_handle:
        # Meeting handle should test calendar event creation
        text_content_for_test = "Please schedule a meeting titled 'Project Kickoff' for January 5th, 2024, at 2:00 PM EST to discuss the project milestones. My email is sender.test@example.com and please invite colleague@example.com."
        # Add a dummy file if process_attachments is True for meeting
        if handle_instructions.process_attachments:
            attachments_for_test = [("meeting_context.txt", b"Meeting context document.", "text/plain")]

    email_data, email_attachments_dir_str, attachment_info = prepare_email_request_data(
        to_email=to_email_address,
        subject=subject_for_test,
        text_content=text_content_for_test,
        attachments_data=attachments_for_test,
    )

    # Set SKIP_EMAIL_DELIVERY for specific handles if they don't typically result in a direct reply "sent"
    # or to simplify testing by not needing to validate a complex LLM-generated reply.
    # For now, assume all handles in DEFAULT_EMAIL_HANDLES are expected to try to send a reply.
    expect_reply_actually_sent = True

    # For "meeting" handle, we will mock send_reply but also check if an ICS was generated.
    # For other handles, we mainly check if a reply was attempted.
    # Note: Since EmailAgent is not mocked beyond EmailSender, the actual content of the reply
    # will depend on the LLM and the prompt templates. This test primarily verifies
    # that the pipeline for each handle type runs and attempts a reply.

    with (
        patch("mxtoai.tasks.EmailSender") as mock_email_sender,
        patch.dict(os.environ, {"SKIP_EMAIL_DELIVERY": ""}),
    ):  # Ensure SKIP_EMAIL_DELIVERY is not set globally for this test run
        mock_sender_instance = mock_email_sender.return_value

        async def mock_async_send_reply(*args, **kwargs):
            # Check for ICS attachment if it's the meeting handle
            if is_meeting_handle:
                sent_attachments = kwargs.get("attachments", [])
                assert any(att["filename"] == "invite.ics" for att in sent_attachments), (
                    "ICS attachment was not prepared for sending by the meeting handle."
                )
                assert any(att["mimetype"] == "text/calendar" for att in sent_attachments)

            # Check for PDF attachment if it's the PDF handle
            if handle_instructions.handle == "pdf":
                sent_attachments = kwargs.get("attachments", [])
                assert len(sent_attachments) > 0, "PDF handle should have attachments"
                pdf_attachment = None
                for att in sent_attachments:
                    if att["filename"].endswith(".pdf"):
                        pdf_attachment = att
                        break
                assert pdf_attachment is not None, "PDF attachment was not prepared for sending by the PDF handle"
                assert pdf_attachment["mimetype"] == "application/pdf", "PDF attachment should have correct mimetype"
                assert len(pdf_attachment["content"]) > 0, "PDF attachment should have content"

            return {"MessageId": "mocked_message_id_happy_path", "status": "sent"}

        mock_sender_instance.send_reply = MagicMock(side_effect=mock_async_send_reply)

        returned_result = process_email_task.fn(
            email_data=email_data, email_attachments_dir=email_attachments_dir_str, attachment_info=attachment_info
        )

        if expect_reply_actually_sent:
            mock_sender_instance.send_reply.assert_called_once()
            _assert_basic_successful_processing(
                returned_result,
                expected_handle=handle_instructions.handle,
                attachments_cleaned_up_dir=email_attachments_dir_str
                if attachments_for_test
                else None,  # Only check cleanup if dir was used
            )
            # Specific assertion for meeting handle's calendar_data
            if is_meeting_handle:
                assert returned_result.calendar_data is not None, "Calendar data should be present for meeting handle"
                assert returned_result.calendar_data.ics_content is not None, (
                    "ICS content should not be None for meeting handle"
                )
                assert len(returned_result.calendar_data.ics_content) > 0, (
                    "ICS content is missing or empty for meeting handle"
                )

            # Specific assertions for PDF handle
            if handle_instructions.handle == "pdf":
                assert returned_result.pdf_export is not None, "PDF export result should be present for PDF handle"
                assert returned_result.pdf_export.filename.endswith(".pdf"), "PDF export should have .pdf filename"
                assert returned_result.pdf_export.file_size > 0, "PDF export should have non-zero file size"
                assert returned_result.pdf_export.mimetype == "application/pdf", (
                    "PDF export should have correct mimetype"
                )
                assert returned_result.pdf_export.title is not None, "PDF export should have a title"
                assert returned_result.pdf_export.pages_estimated >= 1, (
                    "PDF export should have at least 1 page estimated"
                )
        else:
            # If not expecting a sent reply (e.g. if we were to use SKIP_EMAIL_DELIVERY for some handles)
            mock_sender_instance.send_reply.assert_not_called()
            assert returned_result.metadata.email_sent.status == "skipped"  # or "pending" depending on logic
            # Further assertions for non-sent cases might be needed.

        # General check: No errors in metadata for any handle type on happy path
        assert not returned_result.metadata.errors, (
            f"Handle '{handle_instructions.handle}' produced errors: {returned_result.metadata.errors}"
        )

        # Ensure deep_research_mandatory flag was respected (qualitative check via EmailAgent logs if verbose, hard to assert directly without deeper mocks)
        # We trust EmailAgent tests for this part.
        # Here, we are testing the task's integration with the agent for each handle.


# --- PDF Export Specific Tests ---
def test_pdf_export_tool_direct():
    """Test the PDFExportTool directly to ensure it generates valid PDFs."""
    from mxtoai.tools.pdf_export_tool import PDFExportTool

    tool = PDFExportTool()

    # Test basic content export
    result = tool.forward(
        content="# Test Document\n\nThis is a test document with some content.\n\n- Item 1\n- Item 2\n- Item 3",
        title="Test PDF Document",
    )

    assert result["success"] is True, f"PDF export failed: {result.get('error', 'Unknown error')}"
    assert result["filename"] == "Test_PDF_Document.pdf"
    assert result["file_size"] > 0
    assert result["mimetype"] == "application/pdf"
    assert result["title"] == "Test PDF Document"
    assert result["pages_estimated"] >= 1

    # Verify the file actually exists and has content
    pdf_path = result["file_path"]
    assert Path(pdf_path).exists(), "PDF file should exist"

    # Read and verify PDF content
    with Path(pdf_path).open("rb") as f:
        pdf_content = f.read()
    assert len(pdf_content) > 100, "PDF should have substantial content"
    assert pdf_content[:4] == b"%PDF", "File should start with PDF magic bytes"

    # Clean up
    Path(pdf_path).unlink()


def test_pdf_export_tool_with_research_findings():
    """Test PDF export with research findings and attachments summary."""
    from mxtoai.tools.pdf_export_tool import PDFExportTool

    tool = PDFExportTool()

    result = tool.forward(
        content="# Email Newsletter Summary\n\nThis is the main content of the email.",
        title="Weekly Newsletter Export",
        research_findings="## Research Results\n\n1. Finding one\n2. Finding two\n3. Finding three",
        attachments_summary="## Attachments Processed\n\n- attachment1.pdf\n- attachment2.docx",
        include_attachments=True,
    )

    assert result["success"] is True
    assert result["filename"] == "Weekly_Newsletter_Export.pdf"
    assert result["file_size"] > 0

    # Verify the file exists
    pdf_path = result["file_path"]
    assert Path(pdf_path).exists(), "PDF file should exist"

    # Clean up
    Path(pdf_path).unlink()


def test_pdf_export_content_cleaning():
    """Test that PDF export properly cleans email headers and formats content."""
    from mxtoai.tools.pdf_export_tool import PDFExportTool

    tool = PDFExportTool()

    # Content with email headers that should be removed
    email_content = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 01 Jan 2024 12:00:00 +0000

# Important Newsletter

This is the actual content that should be preserved.

## Section 1
- Important point 1
- Important point 2

## Section 2
More important content here.
"""

    result = tool.forward(content=email_content)

    assert result["success"] is True
    assert result["file_size"] > 0

    # The cleaned content should not contain email headers
    # We can't easily verify the internal content without parsing the PDF,
    # but we can verify the tool runs successfully

    # Clean up
    Path(result["file_path"]).unlink()


def test_pdf_handle_full_integration():
    """Test the full PDF handle integration with a more comprehensive email."""
    import shutil
    import tempfile
    from unittest.mock import MagicMock, patch

    from mxtoai.tasks import process_email_task

    # Create comprehensive test email content
    email_data = {
        "to": "pdf@mxtoai.com",
        "from_email": "test@example.com",
        "subject": "Weekly AI Newsletter - Export to PDF",
        "textContent": """# Weekly AI Newsletter

## Top Stories This Week

1. **Breaking**: New AI model achieves breakthrough in natural language understanding
   - Improved accuracy by 15% over previous models
   - Reduced computational requirements
   - Available for public research

2. **Industry News**: Major tech companies announce AI partnerships
   - Focus on ethical AI development
   - Shared research initiatives
   - Open source commitments

3. **Research Highlights**: Recent papers in machine learning
   - Novel architectures for transformer models
   - Advances in computer vision
   - Multimodal learning approaches

## Tools and Resources

- New dataset for training language models
- Updated frameworks and libraries
- Community challenges and competitions

## Upcoming Events

- AI Conference 2024 (March 15-17)
- Workshop on Ethical AI (April 2)
- Research symposium (April 20)

This newsletter provides a comprehensive overview of recent developments in AI research and industry trends.
""",
        "messageId": "<test-pdf-message-id>",
        "date": "2024-01-01T12:00:00Z",
        "recipients": ["pdfå@mxtoai.com"],
        "cc": None,
        "bcc": None,
        "references": None,
        "attachments": [],
    }

    # Create temporary directory for attachments
    temp_dir = tempfile.mkdtemp()

    try:
        with (
            patch("mxtoai.tasks.EmailSender") as mock_email_sender_class,
        ):
            mock_sender_instance = mock_email_sender_class.return_value

            # Track the PDF attachment that gets sent
            captured_pdf_attachment = None

            async def mock_async_send_reply(*args, **kwargs):
                nonlocal captured_pdf_attachment
                sent_attachments = kwargs.get("attachments", [])

                # Find the PDF attachment
                for att in sent_attachments:
                    if att["filename"].endswith(".pdf"):
                        captured_pdf_attachment = att
                        break

                return {"MessageId": "test-pdf-message-id", "status": "sent"}

            mock_sender_instance.send_reply = MagicMock(side_effect=mock_async_send_reply)

            # Run the task
            result = process_email_task.fn(email_data=email_data, email_attachments_dir=temp_dir, attachment_info=[])

            # Verify successful processing
            assert isinstance(result, DetailedEmailProcessingResult)
            # Note: email status will be 'skipped' because test@example.com is in SKIP_EMAIL_DELIVERY
            assert result.metadata.email_sent.status == "skipped"
            assert len(result.metadata.errors) == 0, f"Processing errors: {result.metadata.errors}"

            # Verify PDF export result
            assert result.pdf_export is not None, "PDF export result should be present"
            assert result.pdf_export.filename.endswith(".pdf")
            assert result.pdf_export.file_size > 1000, "PDF should be reasonably sized for the content"
            assert result.pdf_export.mimetype == "application/pdf"
            assert "Weekly AI Newsletter" in result.pdf_export.title or "AI Newsletter" in result.pdf_export.title
            assert result.pdf_export.pages_estimated >= 1

            # Email should not be attempted to be sent due to SKIP_EMAIL_DELIVERY
            mock_sender_instance.send_reply.assert_not_called()

            # The PDF was generated but not attached due to skipped email delivery
            # In this test case, we're focusing on verifying that PDF export works
            # and that the result contains the expected PDF data

            # Verify email response content
            assert result.email_content is not None
            assert result.email_content.text is not None
            assert len(result.email_content.text) > 0
            assert result.email_content.html is not None
            assert len(result.email_content.html) > 0

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pdf_export_error_handling():
    """Test PDF export tool error handling for invalid inputs."""
    from mxtoai.tools.pdf_export_tool import MAX_FILENAME_LENGTH, PDFExportTool

    tool = PDFExportTool()

    # Test with empty content
    result = tool.forward(content="")
    assert result["success"] is True  # Should still work with empty content
    assert result["title"] == "Document"  # Should use default title

    # Clean up if file was created
    if "file_path" in result and Path(result["file_path"]).exists():
        Path(result["file_path"]).unlink()

    # Test with very long title
    long_title = "A" * 200  # Very long title
    result = tool.forward(content="Test content", title=long_title)
    assert result["success"] is True
    # Filename should be truncated - use constant instead of magic number
    max_filename_length = MAX_FILENAME_LENGTH + len(".pdf")  # Reflects truncation logic
    assert len(result["filename"]) <= max_filename_length

    # Clean up
    if "file_path" in result and Path(result["file_path"]).exists():
        Path(result["file_path"]).unlink()


def test_pdf_export_cleanup():
    """Test that PDF export properly cleans up temporary directories."""
    import tempfile

    from mxtoai.tools.pdf_export_tool import PDFExportTool

    # Track temporary directories created
    original_mkdtemp = tempfile.mkdtemp
    created_temp_dirs = []

    def tracking_mkdtemp(*args, **kwargs):
        temp_dir = original_mkdtemp(*args, **kwargs)
        created_temp_dirs.append(temp_dir)
        return temp_dir

    # Patch mkdtemp to track directory creation
    tempfile.mkdtemp = tracking_mkdtemp

    try:
        tool = PDFExportTool()

        # Verify temp directory was created
        assert len(created_temp_dirs) == 1, "Expected exactly one temp directory to be created"
        temp_dir_path = created_temp_dirs[0]
        assert Path(temp_dir_path).exists(), "Temp directory should exist after tool initialization"

        # Generate a PDF
        result = tool.forward(
            content="# Test PDF Cleanup\n\nThis tests that temporary directories are properly cleaned up.",
            title="Cleanup Test PDF",
        )

        assert result["success"] is True, f"PDF generation failed: {result.get('error', 'Unknown error')}"
        pdf_file_path = result["file_path"]
        assert Path(pdf_file_path).exists(), "PDF file should exist"

        # Verify the PDF is in the temp directory
        assert pdf_file_path.startswith(temp_dir_path), "PDF should be in the temp directory"

        # Call cleanup explicitly
        tool.cleanup()

        # Verify temp directory is cleaned up
        assert not Path(temp_dir_path).exists(), "Temp directory should be cleaned up after explicit cleanup"
        assert not Path(pdf_file_path).exists(), "PDF file should be cleaned up with the temp directory"

    finally:
        tempfile.mkdtemp = original_mkdtemp

        # Clean up any remaining directories
        for temp_dir in created_temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


# --- New Test for Delete Handle ---
def test_process_email_task_delete_handle(prepare_email_request_data):
    """
    Tests the delete email handle that removes scheduled tasks.
    Uses actual test database to verify task deletion.
    """
    import uuid

    # First, create a task in the test database to delete
    task_id = uuid.uuid4()
    db_connection = init_db_connection()

    with db_connection.get_session() as session:
        # Create a test task owned by the sender
        test_task = Tasks(
            task_id=task_id,
            email_id="sender.test@example.com",  # Same as the delete requester
            cron_expression="0 9 * * 1",  # Every Monday at 9 AM
            scheduler_job_id=f"job_{task_id}",
            status=TaskStatus.ACTIVE,
            email_request={"from_email": "sender.test@example.com", "subject": "Test Task"},
        )
        session.add(test_task)
        session.commit()

        # Verify task was created
        created_task = session.exec(select(Tasks).where(Tasks.task_id == task_id)).first()
        assert created_task is not None, "Test task should be created"
        assert created_task.status == TaskStatus.ACTIVE, "Test task should be active"

    # Create an email with a delete task request
    delete_content = f"Delete scheduled task {task_id}"

    email_data, email_attachments_dir_str, attachment_info = prepare_email_request_data(
        to_email="delete@mxtoai.com",
        subject="Delete Scheduled Task",
        text_content=delete_content,
    )

    # Mock only the scheduling removal and email sender
    with (
        patch("mxtoai.scheduling.scheduler.Scheduler.remove_job", return_value=True),
        patch("mxtoai.tasks.EmailSender") as mock_email_sender_class,
    ):
        mock_email_sender = mock_email_sender_class.return_value
        mock_email_sender.send_reply = MagicMock()

        async def mock_async_send_reply(*args, **kwargs):
            return {"MessageId": "mocked_message_id_delete", "status": "sent"}

        mock_email_sender.send_reply = MagicMock(side_effect=mock_async_send_reply)

        # Execute the task
        result = process_email_task.fn(
            email_data=email_data,
            email_attachments_dir=email_attachments_dir_str,
            attachment_info=attachment_info,
        )

        # Verify email sender was called
        mock_email_sender.send_reply.assert_called_once()

        # Verify the response
        assert isinstance(result, DetailedEmailProcessingResult)
        assert result.metadata.email_sent.status == "sent"
        assert len(result.metadata.errors) == 0, f"Processing errors: {result.metadata.errors}"

        # Check that the task was actually deleted from the database
        with db_connection.get_session() as session:
            deleted_task = session.exec(select(Tasks).where(Tasks.task_id == task_id)).first()
            assert deleted_task is not None, "Task should still exist in database"
            assert deleted_task.status == TaskStatus.DELETED, "Task should be marked as deleted"

        # Verify the response mentions task deletion
        reply_text = result.email_content.text or ""
        reply_html = result.email_content.html or ""
        assert (
            "delete" in reply_text.lower()
            or "removed" in reply_text.lower()
            or "delete" in reply_html.lower()
            or "removed" in reply_html.lower()
        ), "Response should mention task deletion"
