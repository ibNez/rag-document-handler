"""Tests for email account management."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from email_accounts import EmailAccount, EmailAccountManager


def test_account_crud_operations(tmp_path):
    """Verify add, update, list and remove operations."""
    db = tmp_path / "kb.db"
    manager = EmailAccountManager(str(db))
    acct = EmailAccount(
        name="acct1",
        imap_host="host",
        imap_user="user",
        imap_password="pass",
        imap_port=993,
        mailbox="INBOX",
    )
    manager.add_account(acct)
    accounts = manager.list_accounts()
    assert len(accounts) == 1 and accounts[0].name == "acct1"

    acct.imap_host = "newhost"
    manager.update_account(acct)
    fetched = manager.get_account("acct1")
    assert fetched and fetched.imap_host == "newhost"

    manager.remove_account("acct1")
    assert manager.list_accounts() == []
