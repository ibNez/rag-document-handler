#!/usr/bin/env python3
"""
SSL Certificate Generation Tool for RAG Document Handler

This script generates self-signed SSL certificates for internal use.
For production environments, consider using proper CA-signed certificates.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def create_ssl_directory():
    """Create SSL directory structure."""
    ssl_dir = Path("ssl")
    ssl_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (ssl_dir / "certs").mkdir(exist_ok=True)
    (ssl_dir / "private").mkdir(exist_ok=True)
    (ssl_dir / "config").mkdir(exist_ok=True)
    
    # Set proper permissions (readable by owner only for private keys)
    os.chmod(ssl_dir / "private", 0o700)
    
    return ssl_dir


def create_openssl_config(ssl_dir, hostname="localhost", ip_addresses=None):
    """Create OpenSSL configuration file for certificate generation."""
    if ip_addresses is None:
        ip_addresses = ["127.0.0.1", "::1"]
    
    config_content = f"""[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=US
ST=Internal
L=Enterprise
O=RAG Document Handler
OU=IT Department
CN={hostname}

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = {hostname}
DNS.2 = *.{hostname}
"""
    
    # Add IP addresses
    for i, ip in enumerate(ip_addresses, start=1):
        config_content += f"IP.{i} = {ip}\n"
    
    config_path = ssl_dir / "config" / "openssl.conf"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path


def generate_certificate(ssl_dir, config_path, days=365):
    """Generate SSL certificate and private key."""
    cert_path = ssl_dir / "certs" / "server.crt"
    key_path = ssl_dir / "private" / "server.key"
    
    print(f"Generating SSL certificate...")
    print(f"Certificate: {cert_path}")
    print(f"Private Key: {key_path}")
    
    try:
        # Generate private key and certificate in one command
        cmd = [
            "openssl", "req",
            "-new", "-x509",
            "-days", str(days),
            "-nodes",
            "-keyout", str(key_path),
            "-out", str(cert_path),
            "-config", str(config_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Set proper permissions
        os.chmod(key_path, 0o600)  # Private key readable by owner only
        os.chmod(cert_path, 0o644)  # Certificate readable by all
        
        print("✅ SSL certificate generated successfully!")
        return cert_path, key_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating certificate: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ OpenSSL not found. Please install OpenSSL:")
        print("   macOS: brew install openssl")
        print("   Ubuntu/Debian: sudo apt-get install openssl")
        print("   RHEL/CentOS: sudo yum install openssl")
        sys.exit(1)


def verify_certificate(cert_path):
    """Verify the generated certificate."""
    try:
        cmd = ["openssl", "x509", "-in", str(cert_path), "-text", "-noout"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("\n📋 Certificate Information:")
        print("=" * 50)
        
        # Extract key information
        lines = result.stdout.split('\n')
        for line in lines:
            line = line.strip()
            if "Subject:" in line:
                print(f"Subject: {line.split('Subject: ', 1)[1]}")
            elif "Not Before:" in line:
                print(f"Valid From: {line.split('Not Before: ', 1)[1]}")
            elif "Not After:" in line:
                print(f"Valid Until: {line.split('Not After : ', 1)[1]}")
            elif "DNS:" in line or "IP Address:" in line:
                print(f"Alt Names: {line}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not verify certificate: {e}")


def create_flask_ssl_config(ssl_dir):
    """Create Flask SSL configuration file."""
    cert_path = ssl_dir / "certs" / "server.crt"
    key_path = ssl_dir / "private" / "server.key"
    
    config_content = f"""# SSL Configuration for Flask
# Add these to your .env file or configuration

# SSL Certificate paths
SSL_CERT_PATH={cert_path.absolute()}
SSL_KEY_PATH={key_path.absolute()}

# Flask SSL Context
# In your Flask app, use:
# app.run(host='0.0.0.0', port=3000, ssl_context=(SSL_CERT_PATH, SSL_KEY_PATH))

# For production, consider:
# - Using a reverse proxy (nginx/apache) for SSL termination
# - Using proper CA-signed certificates
# - Implementing HSTS headers
# - Regular certificate rotation
"""
    
    config_file = ssl_dir / "flask_ssl_config.txt"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"\n📝 Flask configuration saved to: {config_file}")
    return config_file


def update_env_example():
    """Update .env.example with SSL configuration."""
    env_example_path = Path(".env.example")
    
    if not env_example_path.exists():
        print("⚠️  .env.example not found, skipping update")
        return
    
    ssl_config = """
# SSL Configuration (Enterprise Internal Use - Always Required)
# HTTPS is always enabled for secure internal communications
SSL_CERT_PATH=ssl/certs/server.crt
SSL_KEY_PATH=ssl/private/server.key

# Update Gmail OAuth redirect URI for HTTPS
# GMAIL_REDIRECT_URI=https://localhost:3000/oauth/gmail/callback
"""
    
    with open(env_example_path, 'a') as f:
        f.write(ssl_config)
    
    print(f"📝 Updated {env_example_path} with SSL configuration")


def main():
    parser = argparse.ArgumentParser(description="Generate SSL certificates for RAG Document Handler")
    parser.add_argument("--hostname", default="localhost", help="Certificate hostname (default: localhost)")
    parser.add_argument("--days", type=int, default=365, help="Certificate validity in days (default: 365)")
    parser.add_argument("--ip", action="append", help="Additional IP addresses for certificate")
    parser.add_argument("--verify", action="store_true", help="Verify existing certificate")
    
    args = parser.parse_args()
    
    print("🔐 RAG Document Handler SSL Certificate Generator")
    print("=" * 50)
    
    # Create SSL directory structure
    ssl_dir = create_ssl_directory()
    
    if args.verify:
        cert_path = ssl_dir / "certs" / "server.crt"
        if cert_path.exists():
            verify_certificate(cert_path)
        else:
            print("❌ No certificate found to verify")
        return
    
    # Prepare IP addresses
    ip_addresses = ["127.0.0.1", "::1"]
    if args.ip:
        ip_addresses.extend(args.ip)
    
    # Create OpenSSL configuration
    config_path = create_openssl_config(ssl_dir, args.hostname, ip_addresses)
    
    # Generate certificate
    cert_path, key_path = generate_certificate(ssl_dir, config_path, args.days)
    
    # Verify certificate
    verify_certificate(cert_path)
    
    # Create Flask configuration
    create_flask_ssl_config(ssl_dir)
    
    # Update .env.example
    update_env_example()
    
    print("\n🎉 SSL setup complete!")
    print("\nNext steps:")
    print("1. Copy SSL settings from ssl/flask_ssl_config.txt to your .env file")
    print("2. Update your Flask application to use SSL context")
    print("3. Update Gmail OAuth redirect URI to use HTTPS")
    print("4. Access your application at: https://localhost:3000")
    print("\n⚠️  Note: Browsers will show a security warning for self-signed certificates.")
    print("   Click 'Advanced' -> 'Proceed to localhost (unsafe)' to continue.")


if __name__ == "__main__":
    main()
