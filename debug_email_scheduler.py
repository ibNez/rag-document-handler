#!/usr/bin/env python3
"""
Email Scheduler Diagnostic Script
Following DEVELOPMENT_RULES.md for all development requirements
"""

import sqlite3
import os
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import socket
from dotenv import load_dotenv

def debug_email_scheduler_issue():
    """Debug why email scheduler isn't picking up emails"""
    
    print("🔍 Email Scheduler Diagnostic")
    print("=" * 40)
    print("Following DEVELOPMENT_RULES.md for all requirements")
    print()
    
    # Load environment
    load_dotenv()
    
    # Check database
    db_path = os.path.join('databases', 'Knowledgebase.db')
    if not os.path.exists(db_path):
        print("❌ Database not found at:", db_path)
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Check email accounts
        print("📧 Checking Email Accounts:")
        cursor.execute("""
            SELECT id, account_name, server_type, server, port, username, 
                   mailbox, refresh_interval_minutes, last_synced
            FROM email_accounts
        """)
        
        accounts = cursor.fetchall()
        if not accounts:
            print("   ❌ No email accounts found in database")
            return
        
        for account in accounts:
            print(f"   Account ID {account[0]}: {account[1]}")
            print(f"      Server: {account[3]}:{account[4]} ({account[2]})")
            print(f"      Username: {account[5]}")
            print(f"      Mailbox: {account[6]}")
            print(f"      Refresh interval: {account[7]} minutes")
            print(f"      Last synced: {account[8]}")
            
            # Check if account is due for sync
            if account[8] is None:
                print("      ✅ Due for sync (never synced)")
            else:
                last_sync = datetime.fromisoformat(account[8])
                next_sync = last_sync + timedelta(minutes=account[7])
                now = datetime.now()
                if now >= next_sync:
                    print(f"      ✅ Due for sync (next: {next_sync}, now: {now})")
                else:
                    time_until = next_sync - now
                    print(f"      ⏱️ Not due yet (in {time_until.total_seconds():.0f} seconds)")
            print()
        
        # 2. Check scheduler configuration
        print("⚙️ Checking Scheduler Configuration:")
        poll_busy = os.getenv('SCHEDULER_POLL_SECONDS_BUSY', '10')
        poll_idle = os.getenv('SCHEDULER_POLL_SECONDS_IDLE', '30')
        print(f"   Poll interval (busy): {poll_busy} seconds")
        print(f"   Poll interval (idle): {poll_idle} seconds")
        
        # 3. Check encryption key
        print("\n🔐 Checking Email Encryption:")
        encryption_key = os.getenv('EMAIL_ENCRYPTION_KEY')
        if not encryption_key:
            print("   ❌ EMAIL_ENCRYPTION_KEY not found in environment")
        else:
            print(f"   ✅ Encryption key found: {encryption_key[:10]}...")
            
            # Try to decrypt a password
            if accounts:
                try:
                    cursor.execute("SELECT password FROM email_accounts WHERE id = ?", (accounts[0][0],))
                    encrypted_password = cursor.fetchone()[0]
                    
                    cipher = Fernet(encryption_key.encode())
                    decrypted = cipher.decrypt(encrypted_password.encode()).decode()
                    print(f"   ✅ Password decryption works (length: {len(decrypted)})")
                except Exception as e:
                    print(f"   ❌ Password decryption failed: {e}")
        
        # 4. Check server connectivity
        print("\n🌐 Checking Email Server Connectivity:")
        for account in accounts:
            server = account[3]
            port = account[4]
            server_type = account[2]
            print(f"   Testing {server}:{port} ({server_type})...")
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((server, port))
                sock.close()
                
                if result == 0:
                    print(f"   ✅ {server}:{port} is reachable")
                    
                    # Check if it's responding with correct protocol
                    if port in [25, 587, 465]:  # SMTP ports
                        print(f"   ⚠️ Port {port} is typically SMTP, not IMAP")
                    elif port in [143, 993]:  # IMAP ports
                        print(f"   ✅ Port {port} is correct for IMAP")
                    else:
                        print(f"   ⚠️ Port {port} is unusual for email")
                        
                else:
                    print(f"   ❌ {server}:{port} is not reachable")
            except Exception as e:
                print(f"   ❌ Connection test failed: {e}")
        
        # 5. Check recent processing status
        print("\n📊 Checking Recent Processing Status:")
        try:
            cursor.execute("""
                SELECT type, status, created_at, details 
                FROM processing_status 
                WHERE type = 'email' 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            
            recent_status = cursor.fetchall()
            if recent_status:
                print("   Recent email processing:")
                for status in recent_status:
                    print(f"   - {status[2]}: {status[1]} ({status[3] or 'no details'})")
            else:
                print("   ❌ No recent email processing status found")
        except sqlite3.OperationalError:
            print("   ⚠️ No processing_status table found")
        
        # 6. Check logs for scheduler activity
        print("\n📝 Checking Recent Logs:")
        log_file = 'rag_document_handler.log'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Look for recent scheduler mentions
            recent_scheduler = []
            recent_email = []
            recent_errors = []
            
            for line in lines[-200:]:  # Last 200 lines
                line_lower = line.lower()
                if 'scheduler' in line_lower or 'heartbeat' in line_lower:
                    recent_scheduler.append(line.strip())
                if 'email' in line_lower and ('ingestion' in line_lower or 'sync' in line_lower):
                    recent_email.append(line.strip())
                if 'error' in line_lower and 'email' in line_lower:
                    recent_errors.append(line.strip())
            
            if recent_scheduler:
                print("   Recent scheduler activity:")
                for line in recent_scheduler[-3:]:  # Last 3
                    print(f"   - {line}")
            else:
                print("   ❌ No recent scheduler activity in logs")
            
            if recent_email:
                print("   Recent email activity:")
                for line in recent_email[-3:]:  # Last 3
                    print(f"   - {line}")
            else:
                print("   ❌ No recent email ingestion activity in logs")
                
            if recent_errors:
                print("   Recent email errors:")
                for line in recent_errors[-3:]:  # Last 3
                    print(f"   - {line}")
        else:
            print("   ❌ Log file not found")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


def check_scheduler_sql_query():
    """Check the specific SQL query used by scheduler"""
    
    print("\n🔍 Checking Scheduler SQL Logic")
    print("=" * 35)
    
    db_path = os.path.join('databases', 'Knowledgebase.db')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # This should match the scheduler's email query logic
        print("Testing scheduler's email due query:")
        
        # Query for emails due for sync (NULL last_synced OR past refresh interval)
        cursor.execute("""
            SELECT id, account_name, last_synced, refresh_interval_minutes,
                   CASE 
                       WHEN last_synced IS NULL THEN 'Never synced - should sync now'
                       WHEN datetime(last_synced, '+' || refresh_interval_minutes || ' minutes') <= datetime('now') 
                            THEN 'Past refresh interval - should sync now'
                       ELSE 'Not due yet'
                   END as sync_status,
                   datetime('now') as current_time,
                   datetime(last_synced, '+' || refresh_interval_minutes || ' minutes') as next_sync_time
            FROM email_accounts
        """)
        
        results = cursor.fetchall()
        
        if results:
            print("Email sync analysis:")
            for result in results:
                print(f"   ID {result[0]} ({result[1]}): {result[4]}")
                print(f"      Current time: {result[5]}")
                if result[2]:
                    print(f"      Last synced: {result[2]}")
                    print(f"      Next sync: {result[6]}")
                    print(f"      Refresh interval: {result[3]} minutes")
                print()
        else:
            print("   ❌ No email accounts found")
        
        # Count emails due
        cursor.execute("""
            SELECT COUNT(*) FROM email_accounts 
            WHERE last_synced IS NULL 
               OR datetime(last_synced, '+' || refresh_interval_minutes || ' minutes') <= datetime('now')
        """)
        
        emails_due = cursor.fetchone()[0]
        print(f"Emails due for sync: {emails_due}")
        
        if emails_due == 0:
            print("❌ This explains why scheduler shows emails_due=0")
            print("   Check if refresh_interval_minutes is too large")
        else:
            print("✅ Scheduler should be processing these emails")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ SQL query check failed: {e}")


def generate_recommendations():
    """Generate specific recommendations based on findings"""
    
    print("\n💡 Diagnostic Recommendations:")
    print("=" * 30)
    
    recommendations = [
        "1. Check if Flask app is running with scheduler enabled",
        "2. Verify EMAIL_ENCRYPTION_KEY is set in .env",
        "3. Ensure email account refresh_interval_minutes is reasonable (1-60)",
        "4. Check email server type (IMAP) and port (143/993) configuration",
        "5. Verify email server is accessible from your network",
        "6. Check application logs for scheduler thread startup",
        "7. Test manual email sync outside of scheduler",
        "8. Verify email account credentials are correct"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")


if __name__ == "__main__":
    debug_email_scheduler_issue()
    check_scheduler_sql_query()
    generate_recommendations()
