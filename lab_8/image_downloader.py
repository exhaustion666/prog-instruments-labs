import aiohttp
import asyncio
import csv
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse


class AsyncImageDownloader:
    def __init__(
        self,
        max_concurrent: int = 20,
        output_dir: str = "downloads",
        log_level: int = logging.INFO
    ) -> None:
        self.max_concurrent = max_concurrent
        self.output_dir = Path(output_dir)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.downloaded_count = 0
        self.failed_count = 0
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    

    def extract_urls_from_csv(self, csv_path: str) -> List[str]:
        urls = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row and row[0].strip():
                        url = row[0].strip()
                        if url.startswith(('http://', 'https://')):
                            urls.append(url)
                        else:
                            self.logger.warning(f"Skipped invalid URL: {url}")
            
            if not urls:
                raise ValueError("CSV file contains no valid URLs")

            return urls
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {csv_path}")
            raise
    

    def get_filename_from_url(self, url: str) -> str:
        parsed_url = urlparse(url)
        path = Path(parsed_url.path)
        
        if path.name:
            filename = path.name
        else:
            filename = f"{parsed_url.netloc}_{hash(url)}.jpg"
        
        safe_filename = "".join(
            c for c in filename if c.isalnum() or c in "._-"
        ).rstrip()
        
        return safe_filename or f"image_{hash(url)}.jpg"
    

    async def download_single_image(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[Path]:
        async with self.semaphore:
            filename = self.get_filename_from_url(url)
            filepath = self.output_dir / filename
            
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        content_type = response.headers.get('Content-Type', '')
                        if not content_type.startswith('image/'):
                            self.logger.warning(
                                f"URL is not an image: {url} "
                                f"(Content-Type: {content_type})"
                            )
                            self.failed_count += 1
                            return None
                        
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        
                        self.downloaded_count += 1
                        return filepath
                    else:
                        self.logger.error(
                            f"HTTP error {response.status} for URL: {url}"
                        )
                        self.failed_count += 1
                        return None
                        
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout downloading: {url}")
                return None
            except aiohttp.ClientError as e:
                self.logger.error(f"Network error for {url}: {str(e)}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error for {url}: {str(e)}")
                return None
    

    async def download_all_images(self, urls: List[str]) -> List[Optional[Path]]:
        self.total_urls = len(urls)
        self.logger.info(f"Starting download of {self.total_urls} images...")
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.download_single_image(session, url)
                for url in urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Exception in task: {str(result)}")
                self.failed_count += 1
            elif result is not None:
                valid_results.append(result)
        
        return valid_results
