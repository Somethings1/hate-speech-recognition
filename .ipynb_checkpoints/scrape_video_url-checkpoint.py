import asyncio
import pandas as pd
from TikTokApi import TikTokApi
from tqdm.asyncio import tqdm_asyncio

# --- Configuration ---
SEED_HASHTAGS = [
    "xuhuong", "vietnam", "giaitri", "hàihước", "learnontiktok",
    "reviewanngon", "goclamdep", "tiktokvietnam", "dance", "fyp"
]
TARGET_URL_COUNT = 3000
VIDEOS_PER_SOURCE = 500

async def main():
    """
    The main asynchronous function to scrape video URLs from TikTok.
    """
    unique_video_urls = set()
    pbar = tqdm_asyncio(total=TARGET_URL_COUNT, desc="Collecting unique video URLs")

    async with TikTokApi() as api:
        print("Creating browser session with correct parameters...")

        # --- START OF THE DEFINITIVE FIX ---
        # Using the 'context_options' argument as specified in the documentation
        # to pass browser context settings like 'ignore_https_errors'.
        await api.create_sessions(
            num_sessions=1,
            headless=True,
            sleep_after=3,
            browser='webkit',
            context_options={
                'ignore_https_errors': True,
            }
        )
        # --- END OF THE DEFINITIVE FIX ---
        
        print("✅ Session created.")

        # --- The rest of the script is identical ---
        print("--- Starting Part 1: Scraping by Hashtag ---")
        for hashtag_name in SEED_HASHTAGS:
            if len(unique_video_urls) >= TARGET_URL_COUNT:
                break
            print(f"\nProcessing hashtag: #{hashtag_name}")
            try:
                hashtag = api.hashtag(name=hashtag_name)
                async for video in hashtag.videos(count=VIDEOS_PER_SOURCE):
                    video_data = video.as_dict
                    author_username = video_data.get("author", {}).get("uniqueId")
                    video_id = video_data.get("id")

                    if author_username and video_id:
                        url = f"https://www.tiktok.com/@{author_username}/video/{video_id}"
                        if url not in unique_video_urls:
                            unique_video_urls.add(url)
                            pbar.update(1)
                        if len(unique_video_urls) >= TARGET_URL_COUNT:
                            break
            except Exception as e:
                print(f"❌ Error processing hashtag '{hashtag_name}': {e}. Moving to next.")
                continue
        
        if len(unique_video_urls) < TARGET_URL_COUNT:
            print("\n--- Target not met. Starting Part 2: Scraping Trending ---")
            try:
                # The region for trending is passed directly to the method.
                async for video in api.trending.videos(count=TARGET_URL_COUNT, region="VN"):
                    video_data = video.as_dict
                    author_username = video_data.get("author", {}).get("uniqueId")
                    video_id = video_data.get("id")

                    if author_username and video_id:
                        url = f"https://www.tiktok.com/@{author_username}/video/{video_id}"
                        if url not in unique_video_urls:
                            unique_video_urls.add(url)
                            pbar.update(1)
                        if len(unique_video_urls) >= TARGET_URL_COUNT:
                            break
            except Exception as e:
                print(f"❌ Error processing trending videos: {e}")

    pbar.close()
    print(f"\n✅ Collection complete. Found {len(unique_video_urls)} unique URLs.")
    
    df_urls = pd.DataFrame(list(unique_video_urls), columns=['video_url'])
    df_urls.to_csv("tiktok_video_urls_final.csv", index=False)
    print("💾 URLs saved to tiktok_video_urls_final.csv")


if __name__ == "__main__":
    asyncio.run(main())