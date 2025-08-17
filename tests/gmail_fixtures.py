import base64
import pytest


@pytest.fixture
def gmail_list_response():
    """Mock response for Gmail API `list` call."""
    return {"messages": [{"id": "m1"}]}


@pytest.fixture
def gmail_get_response():
    """Mock response for Gmail API `get` call containing a raw email with attachment."""
    raw_email = (
        "Subject: Hello\r\n"
        "From: Alice <alice@example.com>\r\n"
        "To: Bob <bob@example.com>\r\n"
        "Date: Mon, 01 Jan 2024 00:00:00 +0000\r\n"
        "Message-ID: <m1@example.com>\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: multipart/mixed; boundary=\"XYZ\"\r\n"
        "\r\n"
        "--XYZ\r\n"
        "Content-Type: text/plain\r\n"
        "\r\n"
        "Hi Bob\r\n"
        "--XYZ\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Disposition: attachment; filename=\"note.txt\"\r\n"
        "\r\n"
        "Attachment content\r\n"
        "--XYZ--\r\n"
    )
    raw_b64 = base64.urlsafe_b64encode(raw_email.encode("utf-8")).decode("utf-8")
    return {"id": "m1", "raw": raw_b64}
