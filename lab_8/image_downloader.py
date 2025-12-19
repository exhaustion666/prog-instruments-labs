import csv
import logging
from pathlib import Path
from typing import List


class AsyncImageDownloader:
    def __init__(
        self,
        max_concurrent: int = 20,
        output_dir: str = "downloads",
        log_level: int = logging.INFO
    ) -> None:
        self.max_concurrent = max_concurrent
        self.output_dir = Path(output_dir)
        
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