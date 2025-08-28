#!/bin/bash

# Install Playwright browsers for URL snapshot functionality
# This script is called during setup to install Playwright browser dependencies

echo "Installing Playwright browser dependencies for URL snapshots..."

# Install Playwright Python package (should already be installed via pip)
echo "Checking Playwright installation..."
python -c "import playwright; print('Playwright Python package found')" || {
    echo "Error: Playwright Python package not found. Please install it first:"
    echo "pip install playwright>=1.40.0"
    exit 1
}

# Install Playwright browsers
echo "Installing Playwright browsers (this may take a few minutes)..."
python -m playwright install chromium

# Verify installation
echo "Verifying Playwright installation..."
python -c "
from playwright.sync_api import sync_playwright
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        print('✓ Playwright Chromium browser installed successfully')
        browser.close()
except Exception as e:
    print(f'✗ Playwright browser verification failed: {e}')
    exit(1)
"

echo "Playwright installation completed successfully!"
echo ""
echo "URL snapshot functionality is now available."
echo "Configure snapshot settings via environment variables:"
echo "  SNAPSHOT_DIR=path/to/snapshots"
echo "  SNAPSHOT_VIEWPORT_WIDTH=1920"
echo "  SNAPSHOT_VIEWPORT_HEIGHT=1080"
echo "  SNAPSHOT_PDF_FORMAT=A4"
echo "  SNAPSHOT_LOCALE=en-US"
echo "Note: Snapshots are always enabled for consistency."
