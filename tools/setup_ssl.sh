#!/bin/bash
"""
SSL Setup Script for RAG Document Handler

This script helps enable SSL for the RAG Document Handler application.
"""

set -e

echo "🔐 RAG Document Handler SSL Setup"
echo "================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Generate SSL certificate if it doesn't exist
if [ ! -f "ssl/certs/server.crt" ] || [ ! -f "ssl/private/server.key" ]; then
    echo "🔑 Generating SSL certificate..."
    python tools/generate_ssl_cert.py --hostname localhost
else
    echo "✅ SSL certificate already exists"
fi

# Update .env file with SSL certificate paths
echo "⚙️  Updating .env configuration for SSL certificate paths..."

# Set certificate paths (SSL is always enabled for enterprise security)
if grep -q "^SSL_CERT_PATH=" .env; then
    sed -i '' 's|^SSL_CERT_PATH=.*|SSL_CERT_PATH=ssl/certs/server.crt|' .env
else
    echo "SSL_CERT_PATH=ssl/certs/server.crt" >> .env
fi

if grep -q "^SSL_KEY_PATH=" .env; then
    sed -i '' 's|^SSL_KEY_PATH=.*|SSL_KEY_PATH=ssl/private/server.key|' .env
else
    echo "SSL_KEY_PATH=ssl/private/server.key" >> .env
fi

# Update Gmail OAuth redirect URI to use HTTPS
if grep -q "^GMAIL_REDIRECT_URI=" .env; then
    sed -i '' 's|^GMAIL_REDIRECT_URI=.*|GMAIL_REDIRECT_URI=https://localhost:3000/oauth/gmail/callback|' .env
else
    echo "GMAIL_REDIRECT_URI=https://localhost:3000/oauth/gmail/callback" >> .env
fi

echo "✅ SSL configuration updated in .env"

echo ""
echo "🎉 SSL setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update your Google OAuth configuration:"
echo "   - Go to Google Cloud Console > APIs & Services > Credentials"
echo "   - Edit your OAuth 2.0 client"
echo "   - Update redirect URI to: https://localhost:3000/oauth/gmail/callback"
echo ""
echo "2. Start the application:"
echo "   source .venv/bin/activate && python main.py"
echo ""
echo "3. Access your application at: https://localhost:3000"
echo ""
echo "⚠️  Note: Your browser will show a security warning for the self-signed certificate."
echo "   Click 'Advanced' → 'Proceed to localhost (unsafe)' to continue."
echo ""
echo "🔍 To verify SSL is working:"
echo "   curl -k https://localhost:3000"
