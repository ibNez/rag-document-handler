# SSL-First Enterprise Configuration

## Overview

The RAG Document Handler now implements an SSL-first approach for enterprise internal use, ensuring encryption in transit and proper OAuth integration from the start.

## Key Changes

### 1. Setup Script Integration (`setup.sh`)
- **Always generates SSL certificates** during setup
- **Automatically configures .env** with SSL settings
- **Updates Gmail OAuth redirect URI** to use HTTPS
- Provides clear instructions for Google OAuth configuration

### 2. Application SSL Support (`rag_manager/app.py`)
- **Automatic SSL context detection** based on configuration
- **HTTP to HTTPS redirect** when SSL is enabled
- **Graceful fallback** with clear logging when certificates are missing

### 3. OAuth Flow Improvements (`rag_manager/web/routes.py`)
- **Always uses HTTPS** for enterprise security
- **No insecure transport allowed** in enterprise deployment
- **Enterprise-ready authentication** flow

### 4. Configuration Management (`rag_manager/core/config.py`)
- **SSL_CERT_PATH**: Path to SSL certificate (required)
- **SSL_KEY_PATH**: Path to SSL private key (required)
- **SSL is always enabled** for enterprise deployment

### 5. Certificate Management (`tools/generate_ssl_cert.py`)
- **Self-signed certificate generation** for internal use
- **Proper permissions** (600 for private keys, 644 for certificates)
- **Subject Alternative Names** for localhost and IP addresses
- **Enterprise certificate details** in generated certificates

### 6. Uninstall Process (`uninstall.sh`)
- **SSL certificate backup** to deleted folder before removal
- **Preserves certificates** for restoration if needed
- **Clean removal** of SSL directory structure

## Enterprise Benefits

### Security Compliance
- ✅ **Encryption in Transit**: All communications use HTTPS
- ✅ **OAuth Compatibility**: No development workarounds needed
- ✅ **Certificate Management**: Proper generation and permissions

### Operational Simplicity
- ✅ **Automated Setup**: SSL configured during installation
- ✅ **Self-Contained**: No external dependencies for certificates
- ✅ **Easy Replacement**: Custom certificates can be substituted

### Development Friendly
- ✅ **Localhost Support**: Works with standard development setup
- ✅ **Browser Compatibility**: Standard self-signed certificate warnings
- ✅ **Debugging Support**: Clear logging for certificate issues

## File Structure

```
ssl/
├── certs/
│   └── server.crt          # SSL certificate (644 permissions)
├── private/
│   └── server.key          # Private key (600 permissions)
├── config/
│   └── openssl.conf        # OpenSSL configuration used for generation
└── flask_ssl_config.txt    # Configuration reference
```

## Configuration Example

```bash
# SSL Configuration (Enterprise Security - Always Required)
# The application always runs with HTTPS for secure enterprise operation.
# Replace with your own certificates if needed by updating the paths below.
SSL_CERT_PATH=ssl/certs/server.crt
SSL_KEY_PATH=ssl/private/server.key
GMAIL_REDIRECT_URI=https://localhost:3000/oauth/gmail/callback
```

## Usage Instructions

### Initial Setup
1. **Run setup script**: `./setup.sh` (automatically generates certificates)
2. **Configure Google OAuth**: Update redirect URI to `https://localhost:3000/oauth/gmail/callback`
3. **Start application**: SSL is enabled by default

### Custom Certificates
1. **Replace generated certificates** with your own files
2. **Update paths** in `.env` if needed
3. **Restart application** to pick up new certificates

### Certificate Regeneration
```bash
python tools/generate_ssl_cert.py --hostname localhost
```

### Verification
```bash
# Check certificate details
openssl x509 -in ssl/certs/server.crt -text -noout

# Test HTTPS connection
curl -k https://localhost:3000

# Application logs show SSL status
```

## OAuth Flow

### Gmail OAuth Configuration
1. **Google Cloud Console**: Update OAuth client redirect URI
2. **Use HTTPS URL**: `https://localhost:3000/oauth/gmail/callback`
3. **No development workarounds**: Standard OAuth flow works

### Authentication Process
1. **User clicks "Authenticate with Gmail"**
2. **HTTPS redirect to Google** (secure transport)
3. **Google redirects back to HTTPS callback** (secure)
4. **Tokens stored securely** with encryption
5. **Account auto-saved** after successful OAuth

## Browser Behavior

### Self-Signed Certificate Warning
- **Expected behavior**: Browser shows security warning
- **User action**: Click "Advanced" → "Proceed to localhost (unsafe)"
- **Enterprise option**: Add certificate to trusted store

### Production Considerations
- **Replace with CA-signed certificates** for production
- **Use reverse proxy** (nginx/apache) for SSL termination
- **Implement HSTS headers** for additional security

## Troubleshooting

### Certificate Issues
- **Missing certificates**: Check `ssl/` directory exists
- **Permission errors**: Verify private key has 600 permissions
- **Generation failures**: Ensure OpenSSL is installed

### OAuth Problems
- **Wrong redirect URI**: Must use HTTPS in Google Console
- **Authentication failures**: Verify certificate is valid
- **SSL certificate missing**: Application requires SSL certificates to start

### Application Startup
- **SSL context errors**: Check certificate and key file paths
- **Port binding issues**: Ensure port 3000 is available
- **Certificate not found**: Run `python tools/generate_ssl_cert.py` to generate certificates

## Migration from HTTP

For existing installations:
1. **Run updated setup.sh** to generate certificates
2. **Update Google OAuth settings** to use HTTPS redirect URI
3. **Restart application** - HTTP requests will redirect to HTTPS
4. **Update bookmarks** to use `https://localhost:3000`

This SSL-first approach ensures the RAG Document Handler meets enterprise security requirements while maintaining ease of use for internal deployment.
