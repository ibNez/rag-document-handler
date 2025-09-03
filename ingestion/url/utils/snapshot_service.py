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
from datetime import datetime, timedelta
from pathlib import Path

# Playwright imports (no fallback as per requirements)
from playwright.async_api import async_playwright
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
    
    def __init__(self, postgres_manager, config, url_manager=None):
        """Initialize snapshot service with database and configuration."""
        self.postgres_manager = postgres_manager
        self.url_manager = url_manager
        self.config = config
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
        Build directory path and filename stem for snapshot.
        
        Returns:
            Tuple of (directory_path, filename_stem)
        """
        parsed = urlparse(url)
        
        # Get domain (punycode normalized for international domains)
        domain = "unknown"
        if parsed.hostname:
            try:
                domain = parsed.hostname.encode("idna").decode("ascii")
            except Exception:
                domain = parsed.hostname.lower()
        
        # Create path slug from URL path
        path_slug = self.slugify(parsed.path or "/")
        
        # Build filename components
        timestamp = self.timestamp_utc()
        query_hash = self.query_string_hash(url)
        variant = "_".join(variant_tokens)
        
        # Build directory and filename stem
        directory = os.path.join(self.snapshot_dir, domain, path_slug)
        filename_stem = f"{timestamp}__q-{query_hash}__v-{variant}"
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        return directory, filename_stem
    
    async def create_snapshot(self, url_id: str, url: str, 
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
                    "pdf_path": existing_snapshot["file_path"],
                    "json_path": existing_snapshot["json_path"],
                    "file_size": existing_snapshot["file_size"],
                    "content_hash": content_hash_value,
                    "final_url": final_url,
                    "http_status": http_status,
                    "message": "Content unchanged - reusing existing snapshot"
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
                "created_at": datetime.utcnow().isoformat() + "Z"
            }
            
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Get file size
            file_size = os.path.getsize(final_pdf_path)
            
            logger.info(f"Successfully created snapshot: {final_pdf_path}")
            
            return {
                "success": True,
                "pdf_path": final_pdf_path,
                "json_path": final_json_path,
                "file_size": file_size,
                "content_hash": content_hash_value,
                "final_url": final_url,
                "http_status": http_status,
                "metadata": metadata
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
    
    def store_snapshot_document(self, url_id: str, url: str, snapshot_result: Dict[str, Any]) -> Optional[str]:
        """
        Store snapshot as a document record in PostgreSQL.
        
        Args:
            url_id: Original URL ID
            url: Original URL
            snapshot_result: Result from create_snapshot()
            
        Returns:
            Document ID if successful, None otherwise
        """
        if not snapshot_result.get("success"):
            logger.error(f"Cannot store document for failed snapshot: {snapshot_result}")
            return None
        
        try:
            # Extract metadata
            pdf_path = snapshot_result["pdf_path"]
            file_size = snapshot_result["file_size"]
            metadata = snapshot_result["metadata"]
            
            # Create descriptive title for the snapshot
            timestamp = metadata["timestamp_utc"]
            parsed_url = urlparse(url)
            domain = parsed_url.hostname or "unknown"
            
            # Use the original URL title or generate one
            url_record = None
            if self.url_manager:
                url_record = self.url_manager.get_url_by_id(url_id)
            original_title = url_record.get("title", domain) if url_record else domain
            
            snapshot_title = f"{original_title} (Snapshot {timestamp})"
            filename = os.path.basename(pdf_path)  # Extract just the PDF filename
            
            # Store in documents table
            document_id = self.postgres_manager.store_document(
                file_path=pdf_path,
                filename=filename,
                title=snapshot_title,
                content_type="text/html",  # Content type remains HTML as per spec
                file_size=file_size,
                document_type="url"
            )
            
            logger.info(f"Stored snapshot document {document_id} for URL {url_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to store snapshot document for URL {url_id}: {e}")
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
            # Build the expected directory structure for this URL
            parsed = urlparse(url)
            domain = "unknown"
            if parsed.hostname:
                try:
                    domain = parsed.hostname.encode("idna").decode("ascii")
                except Exception:
                    domain = parsed.hostname.lower()
            
            path_slug = self.slugify(parsed.path or "/")
            directory = os.path.join(self.snapshot_dir, domain, path_slug)
            
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
                    """, (datetime.utcnow(), document_id))
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
    
    def cleanup_old_snapshots(self, url_id: str) -> Dict[str, Any]:
        """
        Clean up old snapshots based on retention policies.
        
        Args:
            url_id: URL ID to clean up snapshots for
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            # Get URL configuration from database
            url_record = None
            retention_days = None
            max_snapshots = None
            
            try:
                with self.postgres_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT snapshot_retention_days, snapshot_max_snapshots FROM urls WHERE id = %s",
                        (url_id,)
                    )
                    result = cursor.fetchone()
                    if result:
                        retention_days_raw = result.get("snapshot_retention_days")
                        max_snapshots_raw = result.get("snapshot_max_snapshots")
                        
                        # Convert to integers, default to 0 if missing (0 = unlimited)
                        try:
                            retention_days = int(retention_days_raw) if retention_days_raw is not None else 0
                            max_snapshots = int(max_snapshots_raw) if max_snapshots_raw is not None else 0
                            
                            logger.debug(f"Retrieved retention policy for URL {url_id}: {retention_days} days, {max_snapshots} max (0=unlimited)")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Failed to convert retention values to integers: retention_days_raw={retention_days_raw}, max_snapshots_raw={max_snapshots_raw}, error={e}")
                            return {"success": False, "error": f"Invalid retention values in database: {e}"}
                    else:
                        return {"success": False, "error": "URL not found"}
            except Exception as e:
                logger.error(f"Failed to retrieve URL retention policy for {url_id}: {e}")
                return {"success": False, "error": f"Database query failed: {e}"}
            
            # If no retention policies set (both are 0), nothing to clean
            if retention_days == 0 and max_snapshots == 0:
                return {"success": True, "message": "No retention policies set (unlimited)", "deleted": 0}
            
            # Get all snapshot documents for this URL
            snapshot_docs = self._get_snapshot_documents(url_id)
            
            deleted_count = 0
            delete_candidates = []
            
            # Apply retention day policy (0 = unlimited)
            if retention_days > 0:
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                for doc in snapshot_docs:
                    doc_created = datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00"))
                    if doc_created < cutoff_date:
                        delete_candidates.append(doc)
            
            # Apply max snapshots policy (keep most recent, 0 = unlimited)
            if max_snapshots > 0 and len(snapshot_docs) > max_snapshots:
                # Sort by creation date (newest first)
                sorted_docs = sorted(snapshot_docs, 
                                   key=lambda x: x["created_at"], 
                                   reverse=True)
                # Add older snapshots beyond max to delete candidates
                delete_candidates.extend(sorted_docs[max_snapshots:])
            
            # Remove duplicates
            delete_candidates = list({doc["id"]: doc for doc in delete_candidates}.values())
            
            # Delete snapshots
            for doc in delete_candidates:
                if self._delete_snapshot(doc):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old snapshots for URL {url_id}")
            
            return {
                "success": True,
                "deleted": deleted_count,
                "retention_days": retention_days,
                "max_snapshots": max_snapshots
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up snapshots for URL {url_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_snapshot_documents(self, url_id: str) -> List[Dict[str, Any]]:
        """Get all snapshot documents for a URL."""
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Find documents that are snapshots (contain URL ID in title or file_path)
                    cursor.execute("""
                        SELECT id, title, file_path, created_at, file_size
                        FROM documents 
                        WHERE document_type = 'url' 
                        AND (title LIKE %s OR file_path LIKE %s)
                        ORDER BY created_at DESC
                    """, (f"%{url_id}%", f"%{url_id}%"))
                    
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Error getting snapshot documents for URL {url_id}: {e}")
            return []
    
    def _delete_snapshot(self, doc: Dict[str, Any]) -> bool:
        """Delete a snapshot document and its files."""
        try:
            doc_id = doc["id"]
            file_path = doc["file_path"]
            
            # Delete from database
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                    conn.commit()
            
            # Delete PDF file
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete JSON sidecar if it exists
            json_path = file_path.replace(".pdf", ".json")
            if os.path.exists(json_path):
                os.remove(json_path)
            
            logger.debug(f"Deleted snapshot document {doc_id} and files")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting snapshot {doc.get('id')}: {e}")
            return False
