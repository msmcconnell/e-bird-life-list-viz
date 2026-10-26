from ddgs import DDGS
import requests
# query = "Python programming"
# results = DDGS().text(query, max_results=5)
# print(results)

results = DDGS().images(
    query = "blue-bird",
    region = "us-en",
    safesearch = "moderate",
    timelimit = None,
    max_results = 2,
    page = 1,
    backend = "auto",
    size = None,
    color = None,
    type_image = "photo",
    layout = None,
    license_image = None,
)
print(results)


url = results[0]["image"]
local_filename = "downloaded_image.png"
# headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#     }

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com/",  # Bypasses anti-hotlinking protections
    "Connection": "keep-alive"
}
# Send a GET request to the URL
response = requests.get(url, headers=headers)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open a local file in write-binary ('wb') mode
    with open(local_filename, "wb") as file:
        file.write(response.content)
    print("Download complete!")
else:
    print(f"Failed to download file. Status code: {response.status_code}")