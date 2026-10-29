from icrawler.builtin import BingImageCrawler


categories = ['jeans', 'shoes']

for cat in categories:
    print(f"Downloading images for: {cat}...")
    crawler = BingImageCrawler(storage={'root_dir': f'dataset/train/{cat}'})
    crawler.crawl(keyword=cat, max_num=70)
    print(f"✅ {cat} done!")

print("Bhai, saari images download ho gayin!")
