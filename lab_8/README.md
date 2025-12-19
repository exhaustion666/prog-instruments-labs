## About

Asynchronous image downloader from CSV files with parallel download support. The program reads a list of image URLs from a CSV file and downloads them asynchronously with concurrent download limiting. Supports progress logging and error handling.

## Usage

python image_downloader.py images.csv - path to file with images urls

python image_downloader.py images.csv -o my_images -c 10 - the name of your own directory with the maximum number of simultaneous downloads
