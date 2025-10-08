# Gmail OAuth 2.0 Setup Guide

This guide explains how to set up Gmail OAuth 2.0 authentication for the RAG Document Handler email ingestion system.

## Overview

Gmail requires OAuth 2.0 authentication instead of traditional username/password authentication. This provides better security and allows fine-grained permission control.

## Prerequisites

- A Google account
- Access to Google Cloud Console
- The RAG Document Handler application running

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" dropdown at the top
3. Click "New Project"
4. Enter a project name (e.g., "RAG Document Handler")
5. Click "Create"
6. Wait for the project to be created and select it

## Step 2: Enable the Gmail API

1. In the Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Gmail API"
3. Click on "Gmail API" in the search results
4. Click "Enable"
5. Wait for the API to be enabled

## Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" for user type (unless you have Google Workspace)
   - Fill in the required fields:
     - App name: "RAG Document Handler"
     - User support email: Your email
     - Developer contact information: Your email
   - Click "Save and Continue"
   - On the Scopes page, click "Save and Continue" (we'll add scopes later)
   - On the Test Users page, add your email address, then click "Save and Continue"

4. Back on the Credentials page, click "Create Credentials" > "OAuth 2.0 Client IDs"
5. Choose "Web application" as the application type
6. Enter a name (e.g., "RAG Document Handler Web Client")
7. Under "Authorized redirect URIs", add:
   ```
   http://localhost:3000/oauth/gmail/callback
   ```
   (Replace `localhost:3000` with your actual host and port if different)
8. Click "Create"

## Step 4: Download and Configure Credentials

1. After creating the OAuth client, you'll see a dialog with your credentials
2. Click "Download JSON" to download the credentials file
3. Open the downloaded JSON file and extract these values:
   - `client_id`: Found as "client_id" in the JSON
   - `client_secret`: Found as "client_secret" in the JSON
   - `redirect_uris`: One of the URIs you configured (typically the first one)

## Step 5: Configure Environment Variables

Add the following to your `.env` file:

```bash
# Gmail OAuth 2.0 Configuration
GMAIL_CLIENT_ID=your_client_id_here
GMAIL_CLIENT_SECRET=your_client_secret_here
GMAIL_REDIRECT_URI=http://localhost:3000/oauth/gmail/callback
```

Replace the placeholder values with the actual values from your downloaded JSON file.

## Step 6: OAuth Scopes Configuration

**CRITICAL**: The application requires specific OAuth scopes for full email access.

### Required Scopes

For RAG document processing, you **MUST** configure these scopes:

1. **Required Scope**:
   - `https://www.googleapis.com/auth/gmail.readonly` - **REQUIRED** for complete email access
   - This scope provides full email content, headers, body, and attachments

**IMPORTANT**: Do NOT add `gmail.metadata` scope - it's insufficient and creates authorization confusion

### Configure Scopes in Google Cloud Console

**IMPORTANT**: You must explicitly add the required scope to your OAuth consent screen:

1. Go to Google Cloud Console > APIs & Services > **OAuth consent screen**
2. Click "Edit App" 
3. Navigate to the "Scopes" section
4. Click "Add or Remove Scopes"
5. **MANUALLY ADD** this scope (and ONLY this scope):
   ```
   https://www.googleapis.com/auth/gmail.readonly
   ```
6. Save the configuration

### Scope Validation

The Gmail connector includes automatic scope validation:
- ✅ If `gmail.readonly` scope: Full email access available
- ❌ If `gmail.metadata` scope: **Will fail with "Metadata scope doesn't allow format FULL"**

### Why This Matters

- **gmail.metadata**: Only allows reading email headers (sender, subject, date)
- **gmail.readonly**: Allows reading full email content including body and attachments
- **For RAG processing**: Full email content is required for meaningful document extraction

## Step 7: Testing the Setup

1. Restart your RAG Document Handler application
2. Go to the email accounts management page
3. Click "Add Email Account"
4. Select "Gmail" as the server type
5. The form should show OAuth-specific fields instead of password fields
6. Follow the OAuth flow to authenticate with Google

## Security Considerations

- **Keep credentials secure**: Never commit OAuth credentials to version control
- **Use HTTPS in production**: OAuth requires HTTPS for security in production
- **Token storage**: OAuth tokens are encrypted before being stored in the database
- **Token refresh**: The application automatically refreshes expired tokens
- **Scope limitation**: Only request the minimum required scopes

## Troubleshooting

### "Gmail API has not been used in project before or it is disabled"
**Solution**: Enable the Gmail API in Google Cloud Console:
1. Go to [Google Cloud Console > APIs & Services > Library](https://console.developers.google.com/apis/library)
2. Search for "Gmail API" and click "Enable"
3. Wait a few minutes for activation to propagate

### "Metadata scope doesn't allow format FULL"
**Root Cause**: OAuth token only has `gmail.metadata` scope instead of required `gmail.readonly`
**Solution**: 
1. Go to Google Cloud Console > APIs & Services > OAuth consent screen
2. Add the scope: `https://www.googleapis.com/auth/gmail.readonly`
3. Delete the Gmail account from RAG Document Handler
4. Re-add the Gmail account to trigger new OAuth authorization

### "Error 400: redirect_uri_mismatch"
- Ensure the redirect URI in your OAuth configuration exactly matches the one in Google Cloud Console
- Check for trailing slashes or other minor differences
- For HTTPS: Use `https://localhost:3000/oauth/gmail/callback`
- For HTTP: Use `http://localhost:3000/oauth/gmail/callback`

### "Error 403: access_denied"
- Ensure your Google account has access to the Gmail API
- Check that the OAuth consent screen is properly configured
- Verify you're included in "Test Users" if the app is in testing mode

### "Error 401: invalid_client"
- Verify that your client ID and client secret are correct
- Ensure the OAuth client is enabled in Google Cloud Console

### Token Expiration Issues
- The application automatically refreshes tokens, but if you see persistent authentication errors, you may need to re-authenticate
- Check the `oauth_token_expires_at` field in the database

### Development Environment Limitations
Some OAuth applications may have restrictions:
- **Domain restrictions**: OAuth app may be limited to specific domains
- **Testing mode**: Apps in testing mode may have limited scope grants
- **Unverified apps**: Gmail may show warnings for unverified applications

## Database Schema Changes

The `email_accounts` table has been updated to support OAuth authentication:

```sql
ALTER TABLE email_accounts 
ADD COLUMN oauth_access_token TEXT,
ADD COLUMN oauth_refresh_token TEXT,
ADD COLUMN oauth_token_expires_at TIMESTAMP;

-- Make password nullable for OAuth accounts
ALTER TABLE email_accounts ALTER COLUMN password DROP NOT NULL;
```

## Development Notes

- OAuth tokens are stored encrypted in the database using the `EMAIL_ENCRYPTION_KEY`
- The Gmail connector will automatically handle token refresh when needed
- For development, you can use `http://localhost:3000` as the redirect URI
- For production, ensure you use HTTPS and update the redirect URI accordingly

## Additional Resources

- [Google OAuth 2.0 Documentation](https://developers.google.com/identity/protocols/oauth2)
- [Gmail API Documentation](https://developers.google.com/gmail/api)
- [Google Cloud Console](https://console.cloud.google.com/)
