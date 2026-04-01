#!/bin/bash

# Define the file containing the links
LINKS_FILE="./wiki_revisions_urls.txt"
# Define the directory to save the downloaded files
DOWNLOAD_DIR="downloads"
# Define the log file
LOG_FILE="download_log.txt"

# Check if the file exists
if [ ! -f "$LINKS_FILE" ]; then
  echo "File $LINKS_FILE does not exist."
  exit 1
fi

# Create the download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Initialize the log file
echo "Download log - $(date)" > "$LOG_FILE"
echo "=======================" >> "$LOG_FILE"

# Initialize the count
count=0
success_count=0

# Function to download a link and log the result
download_link() {
  local link=$1
  local filename=$(basename "$link")
  if wget -q -O "$DOWNLOAD_DIR/$filename" "$link"; then
    echo "Downloaded: $link" >> "$LOG_FILE"
    success_count=$((success_count+1))
  else
    echo "Failed: $link" >> "$LOG_FILE"
  fi
  count=$((count+1))
}

# Read the links file and process each link
while IFS= read -r link; do
  download_link "$link"
done < "$LINKS_FILE"

# Log the total number of downloaded files
echo "Total downloaded: $success_count out of $count" >> "$LOG_FILE"

echo "Downloads completed and saved to $DOWNLOAD_DIR."
echo "Check $LOG_FILE for details."
