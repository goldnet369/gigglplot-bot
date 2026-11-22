import os, json, random, subprocess, asyncio, time
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.5-flash-lite",
          "gemini-2.5-flash-lite", "gemini-2.5-flash"]
VOICES = ["en-US-AndrewNeural", "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural"]
TOPICS = ["a clumsy cat", "a lazy office worker", "a school prank", "a confusing GPS",
          "a picky eater", "a grandma and technology", "a dog who thinks he is the boss",
          "a bad first date", "a surprise birthday party", "a wrong text message",
          "a very honest kid", "a magic wish that goes wrong"]
PALETTES = [((255, 94, 98), (255, 195, 113)), ((106, 17, 203), (37, 117, 252)),
            ((17, 153, 142), (56, 239, 125)), ((252, 74, 26), (247, 183, 51))]
W, H = 1080, 1920
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def make_story():
    topic = random.choice(TOPICS)
    prompt = (
        f"Write a short funny story about {topic} for a YouTube Short. "
        "Between 90 and 110 words. Simple English, family friendly. "
        "Make it genuinely funny: open with a strong hook in the first sentence, "
        "use short punchy sentences and some dialogue, escalate the absurdity step by step, "
        "and end with a surprising punchline in the very last sentence. "
        "No moral, no cliches, use specific silly details. "
        'Return only JSON like {"title": "...", "story": "...", "description": "...", "tags": ["..."]}. '
        "The title must be curiosity-driven and under 55 characters. "
        "The description is one catchy sentence that teases the story without spoiling the punchline. "
        "The tags are 8 short search keywords."
    )
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}}
    last = None
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(3):
            try:
                r = requests.post(url, params={"key": os.environ["GEMINI_API_KEY"]},
                                  json=body, timeout=90)
                if r.status_code in (404, 400):
                    print("skip model", model, r.status_code)
                    break
                r.raise_for_status()
                data = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
                if isinstance(data, list):
                    data = data[0]
                print("model used:", model)
                tags = [str(t).strip() for t in data.get("tags", []) if str(t).strip()]
                return (data["title"].strip(), data["story"].strip(),
                        str(data.get("description", "")).strip(), tags)
            except Exception as e:
                last = e
                print("retry", model, attempt + 1, e)
                time.sleep(10)
    raise RuntimeError(f"All models failed: {last}")


async def tts(text, path):
    await edge_tts.Communicate(text, random.choice(VOICES), rate="+5%").save(path)


def duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nw=1:nk=1", path])
    return float(out)


def chunks(text, n=6):
    res, cur = [], []
    for w in text.split():
        cur.append(w)
        if len(cur) >= n or (w[-1] in ".!?" and len(cur) >= 3):
            res.append(" ".join(cur))
            cur = []
    if cur:
        res.append(" ".join(cur))
    return res


def wrap(text, font, d, maxw):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def make_bg(c1, c2, w=W, h=H):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        col = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=col)
    return img


def make_frame(bg, text, path):
    img = bg.copy()
    d = ImageDraw.Draw(img)
    d.text((W // 2, 150), "GigglePlot", font=ImageFont.truetype(FONT, 64),
           fill="white", anchor="mm", stroke_width=4, stroke_fill="black")
    big = ImageFont.truetype(FONT, 100)
    lines = wrap(text, big, d, W - 160)
    y = H // 2 - len(lines) * 65 + 65
    for line in lines:
        d.text((W // 2, y), line, font=big, fill="white", anchor="mm",
               stroke_width=8, stroke_fill="black")
        y += 130
    img.save(path)


def make_thumb(title, path="thumb.jpg"):
    tw, th = 1280, 720
    img = make_bg(*random.choice(PALETTES), w=tw, h=th)
    d = ImageDraw.Draw(img)
    d.text((tw // 2, 70), "GigglePlot", font=ImageFont.truetype(FONT, 56),
           fill="white", anchor="mm", stroke_width=4, stroke_fill="black")
    size = 110
    while True:
        font = ImageFont.truetype(FONT, size)
        lines = wrap(title, font, d, tw - 140)
        if len(lines) * (size + 20) <= th - 220 or size <= 60:
            break
        size -= 8
    y = th // 2 + 40 - (len(lines) - 1) * (size + 20) // 2
    for line in lines:
        d.text((tw // 2, y), line, font=font, fill="yellow", anchor="mm",
               stroke_width=8, stroke_fill="black")
        y += size + 20
    img.save(path, quality=90)


def build_video(story):
    asyncio.run(tts(story, "voice.mp3"))
    total = duration("voice.mp3")
    parts = chunks(story)
    chars = sum(len(p) for p in parts)
    bg = make_bg(*random.choice(PALETTES))
    entries = []
    for i, p in enumerate(parts):
        make_frame(bg, p, f"f{i}.png")
        entries.append(f"file 'f{i}.png'\nduration {total * len(p) / chars:.3f}\n")
    entries.append(f"file 'f{len(parts) - 1}.png'\n")
    with open("list.txt", "w") as f:
        f.write("".join(entries))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
                    "-i", "voice.mp3", "-vf", "fps=30,format=yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "out.mp4"], check=True)


def upload(title, story, desc, tags):
    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    yt = build("youtube", "v3", credentials=creds)
    all_tags = []
    size = 0
    for t in tags + ["funny", "shorts", "comedy", "story", "humor"]:
        if t.lower() not in [x.lower() for x in all_tags] and size + len(t) < 400:
            all_tags.append(t)
            size += len(t) + 1
    hashtags = " ".join("#" + t.replace(" ", "") for t in all_tags[:5])
    description = (f"{desc}\n\n{story}\n\nNew funny stories every day. "
                   f"Subscribe to GigglePlot!\n\n#shorts {hashtags}")
    body = {
        "snippet": {
            "title": title[:88] + " #shorts",
            "description": description[:4900],
            "tags": all_tags,
            "categoryId": "23",
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload("out.mp4", mimetype="video/mp4", resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]
    print("Uploaded:", vid)
    try:
        make_thumb(title)
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(
            "thumb.jpg", mimetype="image/jpeg")).execute()
        print("Thumbnail set")
    except Exception as e:
        print("Thumbnail failed:", e)


if __name__ == "__main__":
    title, story, desc, tags = make_story()
    print(title)
    print(story)
    build_video(story)
    upload(title, story, desc, tags)
