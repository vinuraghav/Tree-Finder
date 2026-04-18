from bing_image_downloader import downloader

tree_classes = [
    "Neem tree branches"
]

print("Starting the Scrapper")

for tree in tree_classes:
    downloader.download(
        tree,
        limit=100,
        output_dir="tree_dataset",
        verbose=True,
        timeout=60,
        force_replace=False
    )
    print(f"Completed {tree}, moving to the next")

print("Finished Scraping")