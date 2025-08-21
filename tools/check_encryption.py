import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_encryption():
    connection = None
    cursor = None
    try:
        # Check if encryption key is loaded
        encryption_key = os.getenv("EMAIL_ENCRYPTION_KEY")
        print(f"Encryption key loaded: {'Yes' if encryption_key else 'No'}")
        if encryption_key:
            print(f"Encryption key length: {len(encryption_key)} characters")
            print(f"Encryption key: {encryption_key}")

        # Connect to your PostgreSQL database using environment variables
        connection = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT")
        )

        cursor = connection.cursor()

        # Execute the query to see the actual stored password format
        query = sql.SQL("SELECT id, name, username, password FROM email_accounts;")
        cursor.execute(query)

        # Fetch the result
        results = cursor.fetchall()
        if results:
            for row in results:
                account_id, name, username, password = row
                print(f"\nAccount {account_id}: {name}")
                print(f"  Username: {username}")
                print(f"  Password (raw from DB): {password}")
                print(f"  Password length: {len(password) if password else 0}")
                if password:
                    print(f"  Password starts with 'gAAAAAB' (Fernet encrypted): {password.startswith('gAAAAAB')}")
                    print(f"  Password is plaintext-like: {len(password) < 50 and not password.startswith('gAAAAAB')}")
        else:
            print("No email accounts found.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    check_encryption()
