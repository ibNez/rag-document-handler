"""
URL Snapshot Service for RAG Document Handler
Creates PDF snapshots of web pages for historical preservation and reference.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import os
import re
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode
from datetime import datetime, timedelta, timezone
from pathlib import Path

from playwright.async_api import async_playwright

from rag_manager.data.document_data import DocumentDataManager
from playwright.sync_api import sync_playwright, Browser, Page, Response

logger = logging.getLogger(__name__)

# Tracker keys to filter out of query strings for consistent hashing
TRACKER_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "igshid", "mc_eid", "mc_cid", "msclkid", "twclid"
}


class URLSnapshotService:
    """
    Service for creating PDF snapshots of URLs with systematic naming and retention policies.
    
    Implements the naming convention:
    snapshots/<domain>/<path_slug>/<timestampZ>__q-<qs8>__v-<variant>__c-<c8>.pdf
    """
    
    def __init__(self, postgres_manager, config, url_manager=None, milvus_manager=None):
        """Initialize snapshot service with database and configuration."""
        self.postgres_manager = postgres_manager
        self.config = config
        self.document_data_manager = DocumentDataManager(postgres_manager, config=config) if postgres_manager else None
        self.url_manager = url_manager
        self.milvus_manager = milvus_manager  # Add milvus_manager for embedding operations
        self.snapshot_dir = getattr(config, 'SNAPSHOT_DIR', os.path.join('uploaded', 'snapshots'))
        
        # Default snapshot settings from config
        self.default_viewport = (
            getattr(config, 'SNAPSHOT_VIEWPORT_WIDTH', 1920),
            getattr(config, 'SNAPSHOT_VIEWPORT_HEIGHT', 1080)
        )
        self.default_locale = getattr(config, 'SNAPSHOT_LOCALE', "en-US")
        self.default_pdf_format = getattr(config, 'SNAPSHOT_PDF_FORMAT', "A4")
        self.default_emulate_media = getattr(config, 'SNAPSHOT_EMULATE_MEDIA', "print")
        
        # Ensure snapshot directory exists
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        logger.info(f"URLSnapshotService initialized with snapshot_dir: {self.snapshot_dir}")
    
    def slugify(self, text: str, max_len: int = 60) -> str:
        """Convert text to a safe filesystem slug."""
        if not text:
            return "index"
        
        # Replace non-alphanumeric characters with hyphens
        slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
        
        # Truncate and ensure it doesn't end with a hyphen
        slug = (slug[:max_len]).rstrip("-") or "index"
        
        return slug
    
    def query_string_hash(self, url: str) -> str:
        """Generate hash of normalized query string, filtering tracking parameters."""
        parsed = urlparse(url)
        query = parsed.query
        
        if not query:
            return "none"
        
        # Parse query parameters and filter out tracking keys
        pairs = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True)
                if k not in TRACKER_KEYS]
        
        if not pairs:
            return "none"
        
        # Sort for consistent hashing
        pairs.sort(key=lambda kv: (kv[0], kv[1]))
        normalized_query = urlencode(pairs)
        
        return hashlib.sha1(normalized_query.encode("utf-8")).hexdigest()[:8]
    
    def content_hash(self, text: str) -> str:
        """Generate hash of normalized content for deduplication."""
        # Normalize whitespace to reduce spurious changes
        normalized = re.sub(r"\s+", " ", text).strip()
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    
    def timestamp_utc(self) -> str:
        """Generate UTC timestamp in ISO format for filename."""
        return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    
    def build_snapshot_paths(self, url: str, variant_tokens: List[str]) -> Tuple[str, str]:
        """
        Build directory path and filename stem for snapshot using canonical folder logic.
        Returns:
            Tuple of (directory_path, filename_stem)
        """
        from rag_manager.data.url_data import URLDataManager
        # Use canonical folder logic for directory
        directory = URLDataManager.get_snapshot_folder_for_url(url, base_dir=self.snapshot_dir)
        # Build filename components
        timestamp = self.timestamp_utc()
        query_hash = self.query_string_hash(url)
        variant = "_".join(variant_tokens)
        filename_stem = f"{timestamp}__q-{query_hash}__v-{variant}"
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        return directory, filename_stem
    
    async def create_snapshot_files(self, url_id: str, url: str, 
                       viewport: Optional[Tuple[int, int]] = None,
                       locale: Optional[str] = None,
                       pdf_format: Optional[str] = None,
                       emulate_media: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a PDF snapshot of a URL.
        
        Args:
            url_id: UUID of the URL in the database
            url: The URL to snapshot
            viewport: Browser viewport size (width, height)
            locale: Browser locale
            pdf_format: PDF page format (A4, Letter, etc.)
            emulate_media: Media type to emulate (print, screen)
            
        Returns:
            Dictionary with snapshot results and metadata
        """
        try:
            # Use defaults if not specified (ensure non-None values)
            viewport = viewport or self.default_viewport
            locale = locale or self.default_locale
            pdf_format = pdf_format or self.default_pdf_format
            emulate_media = emulate_media or self.default_emulate_media
            
            # Type assertions to help linter
            assert isinstance(locale, str)
            assert isinstance(pdf_format, str)
            assert isinstance(emulate_media, str)
            
            # Build variant tokens for filename
            variant_tokens = [
                f"{emulate_media}{pdf_format}",
                f"vw{viewport[0]}x{viewport[1]}",
                locale
            ]
            
            # Build paths
            directory, filename_stem = self.build_snapshot_paths(url, variant_tokens)
            
            logger.info(f"Creating snapshot for URL {url_id}: {url}")
            
            # Create temporary filename (content hash will be added later)
            temp_pdf_path = os.path.join(directory, filename_stem + "__c-TBD.pdf")
            temp_json_path = os.path.join(directory, filename_stem + "__c-TBD.json")
            
            # Generate PDF with Playwright
            snapshot_result = await self._generate_pdf_with_playwright(
                url, temp_pdf_path, viewport, locale, pdf_format, emulate_media
            )
            
            if not snapshot_result["success"]:
                return snapshot_result
            
            # Get DOM content for hashing
            dom_text = snapshot_result["dom_text"]
            final_url = snapshot_result["final_url"]
            http_status = snapshot_result["http_status"]
            
            # Generate content hash
            content_hash_value = self.content_hash(dom_text)
            
            # Check for existing snapshot with same content hash
            existing_snapshot = self.find_existing_snapshot_with_hash(url, content_hash_value)
            
            if existing_snapshot:
                # Delete the temporary files since we don't need them
                os.remove(temp_pdf_path)
                
                logger.info(f"Found existing snapshot with same content hash {content_hash_value}. "
                           f"Reusing existing file: {existing_snapshot['file_path']}")
                
                return {
                    "success": True,
                    "duplicate": True,
                    "file_path": os.path.dirname(existing_snapshot["file_path"]),
                    "pdf_file": os.path.basename(existing_snapshot["file_path"]),
                    "json_file": os.path.basename(existing_snapshot["json_path"]),
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                }
            
            # Build final filenames with content hash
            final_pdf_path = os.path.join(directory, filename_stem + f"__c-{content_hash_value}.pdf")
            final_json_path = os.path.join(directory, filename_stem + f"__c-{content_hash_value}.json")
            
            # Rename PDF to final name
            os.rename(temp_pdf_path, final_pdf_path)
            
            # Create metadata sidecar JSON
            metadata = {
                "original_url": url,
                "final_url": final_url,
                "http_status": http_status,
                "timestamp_utc": filename_stem.split("__")[0],
                "url_id": url_id,
                "variant": variant_tokens,
                "viewport": {"width": viewport[0], "height": viewport[1]},
                "emulate_media": emulate_media,
                "pdf_options": {"format": pdf_format, "print_background": True},
                "dom_hash8": content_hash_value,
                "query_hash8": self.query_string_hash(url),
                "locale": locale,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
            
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Get file size
            file_size = os.path.getsize(final_pdf_path)
            
            logger.info(f"Successfully created snapshot: {final_pdf_path}")
            
            return {
                "success": True,
                "duplicate": False,
                "file_path": os.path.dirname(final_pdf_path),
                "pdf_file": os.path.basename(final_pdf_path),
                "json_file": os.path.basename(final_json_path),
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            }
            
        except Exception as e:
            logger.error(f"Failed to create snapshot for URL {url_id} ({url}): {e}")
            return {
                "success": False,
                "error": str(e),
                "url_id": url_id,
                "url": url
            }
    
    async def _generate_pdf_with_playwright(self, url: str, pdf_path: str,
                                     viewport: Tuple[int, int], locale: str,
                                     pdf_format: str, emulate_media: str) -> Dict[str, Any]:
        """Generate PDF using Playwright browser automation (async version)."""
        try:
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=True)
                
                # Create context with specified settings
                context = await browser.new_context(
                    locale=locale,
                    viewport={"width": viewport[0], "height": viewport[1]}
                )
                
                # Create page and navigate
                page = await context.new_page()
                
                # Navigate to URL
                response = await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Emulate print media if specified
                if emulate_media == "print":
                    await page.emulate_media(media="print")
                
                # Generate PDF
                await page.pdf(
                    path=pdf_path,
                    format=pdf_format,
                    print_background=True,
                    margin={"top": "0.5in", "right": "0.5in", "bottom": "0.5in", "left": "0.5in"}
                )
                
                # Extract DOM text for content hashing
                dom_text = await page.evaluate("() => document.body.innerText || ''")
                
                # Get final URL and status
                final_url = page.url
                http_status = response.status if response else None
                
                await browser.close()
                
                return {
                    "success": True,
                    "dom_text": dom_text,
                    "final_url": final_url,
                    "http_status": http_status
                }
                
        except Exception as e:
            logger.error(f"Playwright PDF generation failed for {url}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    


    def _find_existing_document(self, url_id: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Find existing document in Postgres for this URL.
        
        Args:
            url_id: Parent URL ID
            url: Current URL being processed
            
        Returns:
            Dictionary with document info if found, None otherwise
        """
        try:
            # Look for documents with same parent_url_id and file_path (URL)
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT d.id as document_id 
                        FROM documents d
                        WHERE d.parent_url_id = %s
                        AND d.document_type = 'url'
                        AND d.file_path = %s
                        ORDER BY d.created_at DESC
                        LIMIT 1
                    """, (url_id, url))
                    
                    result = cursor.fetchone()
                    if result:
                        return dict(result)
                    
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing document for URL {url_id}: {e}")
            return None
    
    def find_existing_snapshot_with_hash(self, url: str, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Find existing snapshot with the same content hash for a URL by checking filesystem.
        
        Args:
            url: The URL to check (used to determine directory structure)
            content_hash: Content hash to match
            
        Returns:
            Dictionary with existing snapshot info, or None if not found
        """
        try:
            # Use the same canonical folder logic as build_snapshot_paths
            from rag_manager.data.url_data import URLDataManager
            directory = URLDataManager.get_snapshot_folder_for_url(url, base_dir=self.snapshot_dir)
            
            if not os.path.exists(directory):
                return None
                
            # Look for files with this content hash in the directory
            for filename in os.listdir(directory):
                if filename.endswith(f"__c-{content_hash}.pdf"):
                    pdf_path = os.path.join(directory, filename)
                    json_path = pdf_path.replace(".pdf", ".json")
                    
                    if os.path.exists(pdf_path):
                        file_size = os.path.getsize(pdf_path)
                        return {
                            "file_path": pdf_path,
                            "json_path": json_path,
                            "file_size": file_size,
                            "content_hash": content_hash
                        }
            
            return None
                    
        except Exception as e:
            logger.error(f"Error finding existing snapshot for URL {url} with hash {content_hash}: {e}")
            return None

    def update_document_timestamp(self, document_id: str) -> bool:
        """
        Update the created_at timestamp of an existing document to current time.
        
        Args:
            document_id: Document ID to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE documents 
                        SET created_at = %s 
                        WHERE id = %s
                    """, (datetime.now(timezone.utc), document_id))
                    conn.commit()
                    
                    logger.info(f"Updated timestamp for document {document_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error updating timestamp for document {document_id}: {e}")
            return False

    def check_content_changed(self, url_id: str, new_content_hash: str) -> bool:
        """
        Check if URL content has changed by comparing content hash.
        
        Args:
            url_id: URL ID to check
            new_content_hash: New content hash to compare
            
        Returns:
            True if content has changed, False otherwise
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT last_content_hash FROM urls WHERE id = %s",
                        (url_id,)
                    )
                    result = cursor.fetchone()
                    
                    if not result:
                        return True  # URL not found, treat as changed
                    
                    last_hash = result.get("last_content_hash")
                    
                    # If no previous hash, content has "changed"
                    if not last_hash:
                        return True
                    
                    # Compare hashes
                    return last_hash != new_content_hash
                    
        except Exception as e:
            logger.error(f"Error checking content change for URL {url_id}: {e}")
            return True  # Err on the side of creating snapshot
    
    def cleanup_old_snapshot_files(self, url: str, retention_days: int = 0, max_snapshots: int = 0) -> Dict[str, Any]:
        """
        Cleanup snapshots to maintain retention requirements for a specific URL.

        Args:
            url: URL string to clean up snapshots for
            retention_days: Number of days to retain snapshots (0 = unlimited)
            max_snapshots: Maximum number of snapshots to keep (0 = unlimited)
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            logger.debug(f"Running cleanup for URL {url}: {retention_days} days, {max_snapshots} max (0=unlimited)")
            
            # If no retention policies set (both are 0), nothing to clean
            if retention_days == 0 and max_snapshots == 0:
                return {"success": True, "message": "No retention policies set (unlimited)", "deleted": 0}
            
            # Step 1: Lookup document record in documents table by file_path (which contains the URL)
            url_document = self._get_document_for_url_by_filepath(url)
            
            if not url_document:
                return {"success": True, "message": "No document found for URL", "deleted": 0}
            
            doc_id = url_document["document_id"]
            logger.debug(f"Found document {doc_id} for URL {url}")
            
            # Step 2: Get all snapshots for this document_id
            snapshots = self._get_snapshots_for_document(doc_id)
            
            if not snapshots:
                return {"success": True, "message": "No snapshots found for URL", "deleted": 0}
            
            logger.debug(f"Found {len(snapshots)} snapshots for document {doc_id}")
            
            # Step 3: Evaluate snapshots against retention requirements
            delete_candidates = []
            
            # Apply retention day policy (0 = unlimited)
            if retention_days > 0:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
                for snapshot in snapshots:
                    created_at = snapshot["created_at"]
                    if isinstance(created_at, str):
                        snapshot_created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        snapshot_created = created_at
                    
                    if snapshot_created < cutoff_date:
                        delete_candidates.append(snapshot)
            
            # Apply max snapshots policy (keep most recent, 0 = unlimited)
            if max_snapshots > 0 and len(snapshots) > max_snapshots:
                def get_sort_key(snapshot):
                    created_at = snapshot["created_at"]
                    if isinstance(created_at, str):
                        return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        return created_at
                
                sorted_snapshots = sorted(snapshots, key=get_sort_key, reverse=True)
                # Add older snapshots beyond max to delete candidates
                for snapshot in sorted_snapshots[max_snapshots:]:
                    delete_candidates.append(snapshot)
            
            # Remove duplicates based on snapshot_id
            seen_snapshots = set()
            unique_candidates = []
            for candidate in delete_candidates:
                snapshot_id = candidate["snapshot_id"]
                if snapshot_id not in seen_snapshots:
                    seen_snapshots.add(snapshot_id)
                    unique_candidates.append(candidate)
            
            logger.debug(f"Identified {len(unique_candidates)} snapshots for deletion")
            
            # Step 4: Delete each snapshot that needs to be culled
            deleted_count = 0
            for snapshot in unique_candidates:
                snapshot_id = snapshot["snapshot_id"]
                logger.debug(f"Deleting snapshot {snapshot_id}")
                
                # Track which cleanup steps completed for this snapshot
                cleanup_steps_completed = []
                
                # Step 4a: Delete embeddings from Milvus documents collection by snapshot_id
                try:
                    if self._delete_snapshot_milvus_embeddings(snapshot_id):
                        cleanup_steps_completed.append("milvus_embeddings")
                        logger.debug(f"Deleted Milvus embeddings for snapshot {snapshot_id}")
                    else:
                        logger.warning(f"Failed to delete Milvus embeddings for snapshot {snapshot_id}")
                except Exception as e:
                    logger.error(f"Error deleting Milvus embeddings for snapshot {snapshot_id}: {e}")
                
                # Step 4b: Delete document_chunks belonging to the snapshot
                try:
                    if self._delete_snapshot_document_chunks(snapshot_id):
                        cleanup_steps_completed.append("document_chunks")
                        logger.debug(f"Deleted document chunks for snapshot {snapshot_id}")
                    else:
                        logger.warning(f"Failed to delete document chunks for snapshot {snapshot_id}")
                except Exception as e:
                    logger.error(f"Error deleting document chunks for snapshot {snapshot_id}: {e}")
                
                # Step 4c: Delete snapshot files from filesystem
                try:
                    if self._delete_snapshot_files_by_name(
                        snapshot.get("file_path"), 
                        snapshot.get("pdf_file"), 
                        snapshot.get("json_file")
                    ):
                        cleanup_steps_completed.append("snapshot_files")
                        logger.debug(f"Deleted snapshot files for snapshot {snapshot_id}")
                    else:
                        logger.warning(f"Failed to delete snapshot files for {snapshot_id}")
                except Exception as e:
                    logger.error(f"Error deleting snapshot files for snapshot {snapshot_id}: {e}")
                
                # Step 4d: Delete snapshot record from url_snapshots table (ALWAYS LAST)
                try:
                    if self._delete_snapshot_record_by_id(snapshot_id):
                        cleanup_steps_completed.append("snapshot_record")
                        deleted_count += 1
                        logger.debug(f"Successfully deleted snapshot record {snapshot_id}")
                        logger.info(f"Completed cleanup for snapshot {snapshot_id}: {', '.join(cleanup_steps_completed)}")
                    else:
                        logger.error(f"Failed to delete snapshot record {snapshot_id} - snapshot will be retried next cycle")
                except Exception as e:
                    logger.error(f"Error deleting snapshot record {snapshot_id}: {e} - snapshot will be retried next cycle")
            
            logger.info(f"Cleaned up {deleted_count} old snapshots for URL {url}")
            
            return {
                "success": True,
                "deleted": deleted_count,
                "retention_days": retention_days,
                "max_snapshots": max_snapshots
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up snapshots for URL {url}: {e}")
            raise

    def _get_documents_for_url(self, url_id: str) -> List[Dict[str, Any]]:
        """
        Get all documents for a URL (where parent_url_id = url_id).
        
        Args:
            url_id: URL ID to get documents for
            
        Returns:
            List of document records with document_id
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id AS document_id
                        FROM documents 
                        WHERE parent_url_id = %s
                        ORDER BY created_at DESC
                    """, (url_id,))
                    
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Error getting documents for URL {url_id}: {e}")
            raise

    def _get_document_for_url_by_filepath(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get the document for a URL by file_path (where file_path = url).
        This is used for discovered URLs that don't have their own url_id.
        
        Args:
            url: URL string to get document for
            
        Returns:
            Document record with document_id, or None if not found
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id AS document_id
                        FROM documents 
                        WHERE file_path = %s AND document_type = 'url'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (url,))
                    
                    result = cursor.fetchone()
                    return dict(result) if result else None
                    
        except Exception as e:
            logger.error(f"Error getting document for URL {url}: {e}")
            raise

    def _get_snapshots_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all snapshots for a document.
        
        Args:
            document_id: Document ID to get snapshots for
            
        Returns:
            List of snapshot records with file paths
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id AS snapshot_id, url_id, document_id, 
                               file_path, pdf_file, json_file, created_at
                        FROM url_snapshots 
                        WHERE document_id = %s
                        ORDER BY created_at DESC
                    """, (document_id,))
                    
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Error getting snapshots for document {document_id}: {e}")
            raise

    def _delete_document_record(self, document_id: str) -> bool:
        """
        Delete a document record from documents table.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                    conn.commit()
                    return cursor.rowcount > 0
                    
        except Exception as e:
            logger.error(f"Error deleting document record {document_id}: {e}")
            raise

    def _get_snapshot_documents(self, url_id: str) -> List[Dict[str, Any]]:
        """
        Get all snapshot documents for a URL with their snapshot information.
        This combines document and snapshot data for cleanup operations with minimal fields.
        
        Args:
            url_id: URL ID to get snapshot documents for
            
        Returns:
            List of combined document/snapshot records with only essential fields
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            d.id AS document_id,
                            d.file_path,
                            d.created_at,
                            us.id AS snapshot_id,
                            us.file_path AS snapshot_file_path,
                            us.pdf_file,
                            us.json_file,
                            us.created_at AS snapshot_created_at
                        FROM documents d
                        JOIN url_snapshots us ON d.id = us.document_id
                        WHERE d.parent_url_id = %s
                        ORDER BY d.created_at DESC
                    """, (url_id,))
                    
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Error getting snapshot documents for URL {url_id}: {e}")
            raise

    def find_orphaned_files(self, url_id: str) -> List[str]:
        """
        Find orphaned snapshot files for a URL (files that exist but have no database records).
        
        Args:
            url_id: URL ID to check for orphaned files
            
        Returns:
            List of orphaned file paths
        """
        try:
            # Get documents for this URL (file_path contains the URL)
            documents = self._get_documents_for_url(url_id)
            if not documents:
                return []
            
            # Use the URL from the first document's file_path to determine directory structure
            url = documents[0]['file_path']
            
            # Get expected directory for this URL
            from rag_manager.data.url_data import URLDataManager
            directory = URLDataManager.get_snapshot_folder_for_url(url, base_dir=self.snapshot_dir)
            
            if not os.path.exists(directory):
                return []
            
            # Get all PDF files in directory
            import glob
            pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
            
            # Get all file paths from snapshot documents for this URL
            snapshot_docs = self._get_snapshot_documents(url_id)
            db_files = {doc['snapshot_file_path'] for doc in snapshot_docs if doc['snapshot_file_path']}
            
            # Find orphaned files
            orphaned_files = [f for f in pdf_files if f not in db_files]
            
            return orphaned_files
            
        except Exception as e:
            logger.error(f"Error finding orphaned files for URL {url_id}: {e}")
            raise

    def find_orphaned_documents(self, url_id: str) -> List[Dict[str, Any]]:
        """
        Find orphaned documents for a URL (documents that have no snapshots).
        
        Args:
            url_id: URL ID to check for orphaned documents
            
        Returns:
            List of orphaned document records
        """
        try:
            documents = self._get_documents_for_url(url_id)
            orphaned_documents = []
            
            for doc in documents:
                snapshots = self._get_snapshots_for_document(doc['document_id'])
                if not snapshots:
                    orphaned_documents.append(doc)
            
            return orphaned_documents
            
        except Exception as e:
            logger.error(f"Error finding orphaned documents for URL {url_id}: {e}")
            raise

    def _delete_snapshot_record_by_id(self, snapshot_id: str) -> bool:
        """
        Delete a single snapshot record from url_snapshots table by snapshot_id.
        
        Args:
            snapshot_id: Snapshot ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM url_snapshots WHERE id = %s", (snapshot_id,))
                    conn.commit()
                    deleted = cursor.rowcount > 0
                    if deleted:
                        logger.debug(f"Deleted snapshot record {snapshot_id}")
                    return deleted
                    
        except Exception as e:
            logger.error(f"Error deleting snapshot record {snapshot_id}: {e}")
            raise

    def _delete_snapshot_document_chunks(self, snapshot_id: str) -> bool:
        """
        Delete document chunks from PostgreSQL by snapshot_id.
        
        Args:
            snapshot_id: ID of the snapshot whose document chunks should be deleted
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.document_data_manager:
                logger.error("No document_data_manager available - cannot delete document chunks")
                return False
            
            # Delete chunks from PostgreSQL document_chunks table by snapshot_id
            deleted_chunks = self.document_data_manager.delete_document_chunks_by_snapshot_id(snapshot_id)
            logger.info(f"Deleted {deleted_chunks} document chunks from PostgreSQL for snapshot {snapshot_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document chunks for snapshot {snapshot_id}: {str(e)}")
            return False

    def _delete_snapshot_milvus_embeddings(self, snapshot_id: str) -> bool:
        """
        Delete embeddings from Milvus by snapshot_id.
        
        Args:
            snapshot_id: ID of the snapshot whose embeddings should be deleted
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.milvus_manager:
                logger.warning(f"No Milvus manager available - skipping embedding deletion for snapshot {snapshot_id}")
                return True
            
            # Delete embeddings from Milvus by snapshot_id directly
            result = self.milvus_manager.delete_by_snapshot_id(snapshot_id)
            if result.get('success', False):
                deleted_count = result.get('deleted_count', 0)
                logger.info(f"Successfully deleted {deleted_count} embeddings for snapshot {snapshot_id} from Milvus")
                return True
            else:
                logger.warning(f"Failed to delete embeddings for snapshot {snapshot_id}: {result.get('error', 'Unknown error')}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting Milvus embeddings for snapshot {snapshot_id}: {str(e)}")
            return False

    def _delete_snapshot_files_by_name(self, file_path: Optional[str], pdf_file: Optional[str], json_file: Optional[str]) -> bool:
        """
        Delete snapshot files from filesystem using provided file information.
        
        Args:
            file_path: Full path to files
            pdf_file: PDF filename
            json_file: JSON filename
            
        Returns:
            True if at least one file was deleted successfully
        """
        try:
            files_deleted = 0
            
            # Construct full paths from directory and filenames
            if file_path and (pdf_file or json_file):
                # Delete PDF file if specified
                if pdf_file:
                    pdf_path = os.path.join(file_path, pdf_file)
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        files_deleted += 1
                        logger.debug(f"Deleted PDF file: {pdf_path}")
                
                # Delete JSON file if specified
                if json_file:
                    json_path = os.path.join(file_path, json_file)
                    if os.path.exists(json_path):
                        os.remove(json_path)
                        files_deleted += 1
                        logger.debug(f"Deleted JSON file: {json_path}")
            
            # If no directory path or filenames provided, cannot delete files
            elif not file_path:
                logger.error("Cannot delete files without directory path")
                raise ValueError("Directory path required for file deletion")
            elif not pdf_file and not json_file:
                logger.error("Cannot delete files without filenames")
                raise ValueError("At least one filename required for file deletion")
            
            if files_deleted == 0:
                logger.warning("No files found to delete")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting snapshot files: {e}")
            raise
