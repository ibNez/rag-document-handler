# Exchange Email Ingestion

This guide covers how to ingest emails from an Exchange server into RAG Knowledgebase Manager.

## Prerequisites

1. **Exchange credentials**
   - Obtain the server URL, email address, and password for the Exchange account.
   - The connector uses the [exchangelib](https://github.com/ecederstrand/exchangelib) library and connects via Exchange Web Services (EWS).
2. **Configure RAG Knowledgebase Manager**
   - Set `EMAIL_ENABLED=true` in your environment to activate email ingestion.
   - Configure `EMAIL_SYNC_INTERVAL_SECONDS` to control how often accounts are synchronized (default `300`).
   - Add an Exchange account to the `email_accounts` table (via UI or SQL) with the following fields:
     - `server_type`: `exchange`
     - `server`: the EWS server hostname
     - `email_address`: your account's primary email address
     - `password`: the account password
     - Optional: `batch_limit` to cap messages fetched per sync

## Minimal Example

```python
from datetime import datetime

from exchangelib import Credentials
from ingestion.email.connector import ExchangeConnector

creds = Credentials(username="user@example.com", password="password")
connector = ExchangeConnector(
    server="exchange.example.com",
    email_address="user@example.com",
    password="password",
)

records = connector.fetch_emails(since_date=datetime(2024, 1, 1))
print(len(records))
```

## Limitations & Scheduling

- **Permissions**: the account must have EWS access enabled.
- **Batch size**: the connector fetches messages in descending order by received date and can be limited with `batch_limit`.
- **Scheduling**: run the `EmailOrchestrator` with `EMAIL_ENABLED=true`, or call `run_email_ingestion` from a cron job at the desired interval.
