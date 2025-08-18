#!/usr/bin/env python3
"""
Manual Email Sync Test
Following DEVELOPMENT_RULES.md for all development requirements
"""

import os
import sys
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

def test_email_ingestion_manually():
    """Test email ingestion outside of scheduler"""
    
    print("🧪 Manual Email Ingestion Test")
    print("=" * 35)
    print("Following DEVELOPMENT_RULES.md for all requirements")
    print()
    
    load_dotenv()
    
    try:
        # Import email ingestion module
        try:
            from email_ingestion import EmailIngestion
            print("✅ EmailIngestion module imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import EmailIngestion: {e}")
            print("   Checking if email_ingestion.py exists...")
            if os.path.exists('email_ingestion.py'):
                print("   ✅ email_ingestion.py file exists")
            else:
                print("   ❌ email_ingestion.py file missing")
            return
        
        # Initialize
        try:
            email_ingestion = EmailIngestion()
            print("✅ EmailIngestion initialized")
        except Exception as e:
            print(f"❌ EmailIngestion initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Get email accounts from database
        db_path = os.path.join('databases', 'Knowledgebase.db')
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, account_name FROM email_accounts LIMIT 1")
        account = cursor.fetchone()
        
        if not account:
            print("❌ No email accounts found for testing")
            return
        
        account_id = account[0]
        account_name = account[1]
        
        print(f"🔧 Testing account: {account_name} (ID: {account_id})")
        
        # Try manual sync
        try:
            print("   Attempting manual sync...")
            result = email_ingestion.sync_account(account_id)
            print(f"✅ Manual sync completed: {result}")
            
            # Update last_synced timestamp
            cursor.execute("""
                UPDATE email_accounts 
                SET last_synced = datetime('now') 
                WHERE id = ?
            """, (account_id,))
            conn.commit()
            print("✅ Updated last_synced timestamp")
            
        except Exception as e:
            print(f"❌ Manual sync failed: {e}")
            import traceback
            traceback.print_exc()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        import traceback
        traceback.print_exc()


def check_email_ingestion_module():
    """Check if email ingestion module and dependencies exist"""
    
    print("\n🔍 Checking Email Ingestion Module")
    print("=" * 35)
    
    # Check if file exists
    if not os.path.exists('email_ingestion.py'):
        print("❌ email_ingestion.py not found")
        return
    
    print("✅ email_ingestion.py exists")
    
    # Check file size and basic content
    with open('email_ingestion.py', 'r') as f:
        content = f.read()
    
    print(f"   File size: {len(content)} characters")
    
    # Look for key classes and methods
    key_items = [
        'class EmailIngestion',
        'def sync_account',
        'def __init__',
        'import imaplib',
        'import email'
    ]
    
    for item in key_items:
        if item in content:
            print(f"   ✅ Found: {item}")
        else:
            print(f"   ❌ Missing: {item}")
    
    # Check for common issues
    if 'def sync_account' in content:
        # Look for the sync_account method
        lines = content.split('\n')
        in_sync_method = False
        sync_method_lines = []
        
        for line in lines:
            if 'def sync_account' in line:
                in_sync_method = True
                sync_method_lines.append(line)
            elif in_sync_method and line.startswith('def '):
                break
            elif in_sync_method:
                sync_method_lines.append(line)
        
        if sync_method_lines:
            print(f"   ✅ sync_account method has {len(sync_method_lines)} lines")
        else:
            print(f"   ❌ sync_account method appears empty")


def check_scheduler_thread():
    """Check if scheduler thread is actually running"""
    
    print("\n🧵 Checking Scheduler Thread")
    print("=" * 25)
    
    # Look for thread information in logs
    log_file = 'rag_document_handler.log'
    if not os.path.exists(log_file):
        print("❌ Log file not found")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Look for thread startup messages
    thread_startup = []
    scheduler_errors = []
    
    for line in lines:
        line_lower = line.lower()
        if 'thread' in line_lower and ('start' in line_lower or 'scheduler' in line_lower):
            thread_startup.append(line.strip())
        if 'scheduler' in line_lower and 'error' in line_lower:
            scheduler_errors.append(line.strip())
    
    if thread_startup:
        print("   Recent thread startup messages:")
        for msg in thread_startup[-3:]:
            print(f"   - {msg}")
    else:
        print("   ❌ No scheduler thread startup messages found")
    
    if scheduler_errors:
        print("   Recent scheduler errors:")
        for error in scheduler_errors[-3:]:
            print(f"   - {error}")
    else:
        print("   ✅ No recent scheduler errors found")


if __name__ == "__main__":
    check_email_ingestion_module()
    check_scheduler_thread()
    test_email_ingestion_manually()
