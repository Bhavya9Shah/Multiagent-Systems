import pandas as pd
import re

raw_path = 'data/original/challenge_videos_final(challenge_videos_final).csv'
df = pd.read_csv(raw_path, encoding='cp1252')

print(f"Dataset shape: {df.shape}")
print("\n--- Data Types & Non-null Counts ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Channel Names & Unique Channel IDs ---")
channel_map = df.groupby('channel_id')['channel_name'].unique().to_dict()
for cid, names in channel_map.items():
    print(f"Channel ID: {cid} -> Names: {names}")

print("\n--- Malformed Video IDs (#NAME? or not 11 chars) ---")
for idx, row in df.iterrows():
    vid = str(row['video_id'])
    url = str(row['thumbnail'])
    cname = str(row['channel_name'])
    if vid.startswith('#') or len(vid) != 11 or not re.match(r'^[A-Za-z0-9_-]{11}$', vid):
        # Extract from URL
        m = re.search(r'/vi/([^/]+)/', url)
        recovered = m.group(1) if m else "UNKNOWN"
        print(f"Row {idx:3d}: raw_id='{vid}' | channel='{cname}' | url='{url}' | recovered_id='{recovered}'")

print("\n--- Duplicate Video IDs Check ---")
dupes = df[df.duplicated(subset=['video_id'], keep=False)]
print(f"Duplicate count by raw video_id: {len(dupes)}")

print("\n--- Face Count Column Summary ---")
print(df['face_count'].describe())
print("Value counts:")
print(df['face_count'].value_counts(dropna=False).sort_index())
