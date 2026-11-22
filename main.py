import os, json, random, subprocess, asyncio
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

MODEL = "gemini-flash-latest"
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
        "Between 90 and 110 words. Simple English, family friendly, "
        "with a funny twist at the end. "
        'Return only JSON like {"title": "...", "story": "..."}. '
        "The title must be under 60 characters."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 1.0}}
    r = requests.post(url, params={"key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=90)
    r.raise_for_status()
    data = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
    if isinstance(data, list):
        data = data[0]
    return data["title"].strip(), data["story"].strip()


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


def make_bg(c1, c2):
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        col = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
        d.line([(0, y), (W, y)], fill=col)
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


def upload(title, story):
    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    yt = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": title[:90] + " #shorts",
            "description": story + "\n\n#shorts #funny #story #comedy",
            "tags": ["funny", "story", "shorts", "comedy", "humor"],
            "categoryId": "23",
        },
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload("out.mp4", mimetype="video/mp4", resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    print("Uploaded:", resp["id"])


if __name__ == "__main__":
    title, story = make_story()
    print(title)
    print(story)
    build_video(story)
    upload(title, story)
