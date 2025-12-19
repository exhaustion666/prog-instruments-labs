import aiohttp
import argparse
import asyncio
import csv
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse


class AsyncImageDownloader:
    """
    Asynchronous image downloader from CSV file.
    
    :param max_concurrent: Maximum number of concurrent downloads
    :param output_dir: Directory to save images
    """
    def __init__(
        self,
        max_concurrent: int = 20,
        output_dir: str = "downloads"
    ) -> None:
        self.max_concurrent = max_concurrent
        self.output_dir = Path(output_dir)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.downloaded_count = 0
        self.failed_count = 0
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    

    def extract_urls_from_csv(self, csv_path: str) -> List[str]:
        """
        Extract image URLs from CSV file.
        
        :param csv_path: Path to CSV file
        :returns: List of URLs
        :raises FileNotFoundError: If file not found
        :raises ValueError: If CSV is empty or has invalid format
        """
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
        """
        Generate filename from URL.
        
        :param url: Image URL
        :returns: Filename with extension
        """
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
        """
        Download single image.
        
        :param session: aiohttp session
        :param url: Image URL
        :returns: Path to saved file or None on error
        """
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
                        self.logger.info(
                            f"Downloaded: {filename} "
                            f"({self.downloaded_count + self.failed_count}/"
                            f"{self.total_urls})"
                        )
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
        """
        Download all images asynchronously.
        
        :param urls: List of URLs
        :returns: List of paths to downloaded files
        """
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
    

    def download_from_csv(self, csv_path: str) -> None:
        """
        Main method to download images from CSV file.
        
        :param csv_path: Path to CSV file
        """
        try:
            urls = self.extract_urls_from_csv(csv_path)
            
            asyncio.run(self.download_all_images(urls))
            
            self.logger.info(
                f"Download completed.\n"
                f"Images saved to: {self.output_dir.absolute()}\n"
            )
            
        except Exception as e:
            self.logger.error(f"Critical error: {str(e)}")
            raise


def main() -> None:
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Asynchronous image download from CSV file"
    )
    parser.add_argument(
        "csv_file",
        help="Path to CSV file with image URLs"
    )
    parser.add_argument(
        "-o", "--output",
        default="downloads",
        help="Directory to save images (default: downloads)"
    )
    parser.add_argument(
        "-c", "--concurrent",
        type=int,
        default=20,
        help="Maximum concurrent downloads (default: 20)"
    )
    
    args = parser.parse_args()
    
    downloader = AsyncImageDownloader(
        max_concurrent=args.concurrent,
        output_dir=args.output
    )
    
    downloader.download_from_csv(args.csv_file)


if __name__ == "__main__":
    main()
