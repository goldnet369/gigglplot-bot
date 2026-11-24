import os, json, math, random, subprocess, asyncio, time, wave
import numpy as np
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.5-flash-lite",
          "gemini-2.5-flash-lite", "gemini-2.5-flash"]
VOICES = ["en-US-AnaNeural", "en-US-JennyNeural"]
TOPICS = ["brushing teeth", "counting to five", "the colors of the rainbow", "a rainy day",
          "clean up time", "bedtime", "a trip to the zoo", "sharing toys", "eating vegetables",
          "washing hands", "the sun and the moon", "making new friends", "dancing",
          "animal sounds", "a big hug", "good morning"]
SKIES = [((120, 200, 255), (255, 240, 170)), ((255, 170, 200), (255, 235, 160)),
         ((150, 130, 255), (130, 230, 255)), ((255, 160, 90), (255, 225, 130))]

SR = 24000
VW, VH, FPS = 720, 1280, 24
BPM = 108
BEAT = 60.0 / BPM
INTRO, OUTRO = 4, 4
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK = (28, 22, 24)
WHITE = (253, 250, 244)


# ---------------------------------------------------------------- song text
def make_song(topic=None):
    topic = topic or random.choice(TOPICS)
    prompt = (
        "Write a very short, cheerful children's song for a YouTube Short, starring a "
        f"friendly character named Giggy. Topic: {topic}. "
        "Exactly 8 lines, each 3 to 7 words, very simple words for kids aged 2 to 6, "
        "strong rhymes, bouncy rhythm. Line 1 and line 5 are the same catchy chorus line. "
        "Safe and positive, nothing scary. "
        'Return only JSON like {"title": "...", "lines": ["..."], "description": "...", "tags": ["..."]}. '
        "The title is under 50 characters and says what the song is about, no clickbait. "
        "The description is one sentence for parents. The tags are 8 short search keywords."
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
                lines = [str(x).strip() for x in data["lines"] if str(x).strip()][:10]
                if len(lines) < 4:
                    raise ValueError("too few lines")
                tags = [str(t).strip() for t in data.get("tags", []) if str(t).strip()]
                print("model used:", model)
                return (data["title"].strip(), lines,
                        str(data.get("description", "")).strip(), tags)
            except Exception as e:
                last = e
                print("retry", model, attempt + 1, e)
                time.sleep(10)
    raise RuntimeError(f"All models failed: {last}")


# -------------------------------------------------------------------- voice
async def tts_line(text, path):
    last = None
    for v in VOICES:
        try:
            await edge_tts.Communicate(text, v, rate="-6%").save(path)
            if os.path.getsize(path) > 1000:
                return
        except Exception as e:
            last = e
    raise RuntimeError(f"TTS failed: {last}")


def decode(path):
    out = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                                   "-ac", "1", "-ar", str(SR), "-"])
    return np.frombuffer(out, dtype=np.float32).copy()


def trim(x):
    idx = np.where(np.abs(x) > 0.02)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(0.03 * SR))
    b = min(len(x), idx[-1] + int(0.05 * SR))
    return x[a:b]


def get_voices(lines):
    voices = []
    for i, line in enumerate(lines):
        p = f"line{i}.mp3"
        asyncio.run(tts_line(line, p))
        v = trim(decode(p))
        peak = float(np.max(np.abs(v))) or 1.0
        voices.append(v / peak * 0.9)
    return voices


# -------------------------------------------------------------------- music
def tone(freq, dur, kind="sine", amp=1.0, decay=5.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * freq * t
    w = np.sin(ph) if kind == "sine" else 2 / np.pi * np.arcsin(np.sin(ph))
    env = np.exp(-decay * t) * np.minimum(1.0, t / 0.006)
    return (w * env * amp).astype(np.float32)


def kick():
    t = np.arange(int(0.25 * SR)) / SR
    ph = 2 * np.pi * (45 * t + 90 * (1 - np.exp(-25 * t)) / 25)
    return (np.sin(ph) * np.exp(-12 * t) * 0.9).astype(np.float32)


def clap(rng):
    t = np.arange(int(0.12 * SR)) / SR
    return (rng.standard_normal(len(t)) * np.exp(-30 * t) * 0.5).astype(np.float32)


def hat(rng):
    t = np.arange(int(0.04 * SR)) / SR
    n = np.diff(rng.standard_normal(len(t) + 1))
    return (n * np.exp(-120 * t) * 0.35).astype(np.float32)


def add(buf, sig, t):
    s = int(t * SR)
    e = min(len(buf), s + len(sig))
    if e > s:
        buf[s:e] += sig[:e - s]


def synth_music(total_beats, seed):
    rng = np.random.default_rng(seed)
    buf = np.zeros(int(total_beats * BEAT * SR) + SR, np.float32)
    roots = [130.81, 98.0, 110.0, 87.31]
    chords = [[261.63, 329.63, 392.0], [392.0, 493.88, 587.33],
              [440.0, 523.25, 659.25], [349.23, 440.0, 523.25]]
    penta = [523.25, 587.33, 659.25, 783.99, 880.0, 1046.5]
    idx, pattern = 2, []
    for _ in range(16):
        idx = int(np.clip(idx + rng.choice([-2, -1, 0, 1, 2]), 0, len(penta) - 1))
        pattern.append(idx if rng.random() < 0.8 else -1)
    k, c, h = kick(), clap(rng), hat(rng)
    for b in range(int(total_beats)):
        bar = (b // 4) % 4
        t = b * BEAT
        add(buf, k, t)
        if b % 4 in (1, 3):
            add(buf, c, t)
        add(buf, h, t)
        add(buf, h * 0.7, t + 0.5 * BEAT)
        add(buf, tone(roots[bar], BEAT * 0.95, "sine", 0.55, 3.0), t)
        for f in chords[bar]:
            add(buf, tone(f, BEAT * 0.4, "tri", 0.10, 9), t + 0.5 * BEAT)
        for half in (0, 1):
            note = pattern[((b % 8) * 2) + half]
            if note >= 0:
                add(buf, tone(penta[note], BEAT * 0.5, "sine", 0.22, 5), t + half * 0.5 * BEAT)
    peak = float(np.max(np.abs(buf))) or 1.0
    return buf / peak


def build_audio(voices, seed):
    slots, beat = [], INTRO
    for v in voices:
        need = len(v) / SR / BEAT + 0.5
        nb = max(4, int(2 * math.ceil(need / 2)))
        slots.append((beat, nb, v))
        beat += nb
    total_beats = beat + OUTRO
    music = synth_music(total_beats, seed)
    voice = np.zeros_like(music)
    for beat, nb, v in slots:
        add(voice, v, beat * BEAT + 0.25 * BEAT)
    mix = music * 0.42 + voice * 0.95
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    with wave.open("mix.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32000).astype(np.int16).tobytes())
    return slots, total_beats, voice


# ---------------------------------------------------------------- character
def draw_giggy(mouth=0.3, hand=0, spark=False):
    S = 1280
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # feet
    for x in (480, 800):
        d.ellipse((x - 85, 1150, x + 85, 1245), fill=WHITE, outline=INK, width=14)
    # body
    d.ellipse((410, 760, 870, 1190), fill=WHITE, outline=INK, width=14)
    # hands
    for i, x in enumerate((345, 935)):
        up = (hand == 0) if i == 0 else (hand == 1)
        y = 800 if up else 940
        d.ellipse((x - 100, y - 100, x + 100, y + 100), fill=WHITE, outline=INK, width=14)
    # head
    cx, cy, rx, ry = 640, 560, 385, 350
    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=WHITE, outline=INK, width=14)
    # cheeks
    for x in (420, 860):
        d.ellipse((x - 70, 580, x + 70, 670), fill=(248, 160, 165))
    # eyes (happy arcs)
    for x in (470, 810):
        d.arc((x - 85, 440, x + 85, 580), 200, 340, fill=INK, width=26)
    # mouth
    my = 610
    if mouth < 0.1:
        d.arc((500, 560, 780, 740), 25, 155, fill=INK, width=20)
    else:
        h = int(40 + 170 * mouth)
        m = Image.new("L", (S, S), 0)
        ImageDraw.Draw(m).pieslice((640 - 140, my - h, 640 + 140, my + h), 0, 180, fill=255)
        layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.pieslice((640 - 140, my - h, 640 + 140, my + h), 0, 180, fill=(70, 15, 28, 255))
        ld.ellipse((640 - 85, my + int(h * 0.45), 640 + 85, my + int(h * 1.4)),
                   fill=(236, 92, 104, 255))
        layer.putalpha(m)
        img.alpha_composite(layer)
        d = ImageDraw.Draw(img)
        d.arc((640 - 140, my - h, 640 + 140, my + h), 0, 180, fill=INK, width=12)
        d.line((640 - 140, my, 640 + 140, my), fill=INK, width=12)
    # cap
    cap = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cap)
    RED, DRED = (225, 32, 40), (170, 20, 28)
    cd.polygon([(870, 250), (1080, 300), (1090, 380), (900, 340)], fill=RED, outline=INK)
    cd.line([(870, 250), (1080, 300), (1090, 380), (900, 340), (870, 250)], fill=INK, width=14, joint="curve")
    cd.pieslice((640 - 350, 70, 640 + 350, 70 + 480), 180, 360, fill=RED, outline=INK, width=14)
    cd.line((640 - 350, 310, 640 + 350, 310), fill=INK, width=14)
    cd.rounded_rectangle((560, 120, 760, 190), 20, fill=DRED, outline=INK, width=8)
    for x in (600, 660, 720):
        cd.ellipse((x - 9, 148, x + 9, 166), fill=(255, 205, 190))
    cap = cap.rotate(-17, resample=Image.BICUBIC, center=(640, 320))
    img.alpha_composite(cap, dest=(0, 38))
    d = ImageDraw.Draw(img)
    # sparkles
    if spark:
        gold = (255, 205, 55)
        for a, b in [((150, 330), (230, 390)), ((95, 470), (200, 480)), ((190, 210), (250, 290)),
                     ((1130, 470), (1030, 480)), ((1090, 600), (1000, 570))]:
            d.line([a, b], fill=gold, width=22)
            for p in (a, b):
                d.ellipse((p[0] - 11, p[1] - 11, p[0] + 11, p[1] + 11), fill=gold)
    return img.resize((S // 2, S // 2), Image.LANCZOS)


# --------------------------------------------------------------------- scene
def sky_frames(pal, n=24):
    top, bot = pal
    base = Image.new("RGB", (VW, VH))
    bd = ImageDraw.Draw(base)
    for y in range(VH):
        t = y / VH
        bd.line([(0, y), (VW, y)], fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    R = 1700
    rays = Image.new("RGBA", (R, R), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rays)
    for i in range(0, 16, 2):
        a0, a1 = i * 22.5, (i + 1) * 22.5
        rd.pieslice((0, 0, R, R), a0, a1, fill=(255, 255, 255, 55))
    frames = []
    for k in range(n):
        ang = k * (45.0 / n)
        rr = rays.rotate(ang, resample=Image.BILINEAR)
        crop = rr.crop(((R - VW) // 2, (R - VH) // 2 - 150, (R + VW) // 2, (R + VH) // 2 - 150))
        f = base.copy()
        f.paste(crop, (0, 0), crop)
        fd = ImageDraw.Draw(f)
        fd.ellipse((-320, 900, 1040, 1700), fill=(98, 205, 118), outline=(60, 150, 85), width=8)
        frames.append(f)
    return frames


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


def layout(text, font, d):
    out, y = [], 0
    for line in wrap(text, font, d, VW - 80):
        words = line.split()
        sp = d.textlength(" ", font=font)
        total = sum(d.textlength(w, font=font) for w in words) + sp * (len(words) - 1)
        x = (VW - total) / 2
        for w in words:
            out.append((w, x, y))
            x += d.textlength(w, font=font) + sp
        y += 78
    return out


def make_video(title, lines):
    voices = get_voices(lines)
    seed = random.randrange(10 ** 6)
    slots, total_beats, voice = build_audio(voices, seed)
    n_frames = int(total_beats * BEAT * FPS)
    spf = SR // FPS
    rms = np.array([np.sqrt(np.mean(voice[i * spf:(i + 1) * spf] ** 2) + 1e-12)
                    for i in range(n_frames)])
    mouth_env, m = [], 0.0
    for r in rms:
        target = min(1.0, (r / 0.16)) ** 0.7
        m = m * 0.45 + target * 0.55
        mouth_env.append(m)

    sky = sky_frames(random.choice(SKIES))
    ft = ImageFont.truetype(FONT, 62)
    fs = ImageFont.truetype(FONT, 44)
    fn = ImageFont.truetype(FONT, 84)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    layouts = {}

    def get_layout(key, text):
        if key not in layouts:
            layouts[key] = layout(text, ft, tmp)
        return layouts[key]

    cache = {}
    notes = [("♪", 90, 0), ("♫", 560, 230), ("♪", 300, 470), ("♫", 640, 700), ("♪", 120, 860)]
    ncols = [(255, 255, 255), (255, 225, 80), (255, 140, 190), (140, 220, 255), (255, 255, 255)]

    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-", "-i", "mix.wav",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-shortest", "out.mp4"],
        stdin=subprocess.PIPE)

    for f in range(n_frames):
        t = f / FPS
        bp = t / BEAT
        phase = bp % 1.0
        img = sky[(f // 2) % len(sky)].copy()
        d = ImageDraw.Draw(img)
        # notes
        for i, (ch, x, off) in enumerate(notes):
            y = 840 - ((t * 70 + off) % 760)
            d.text((x + 14 * math.sin(t * 2 + i), y), ch, font=fn, fill=ncols[i],
                   stroke_width=5, stroke_fill=INK)
        # channel name
        d.text((VW // 2, 62), "GigglePlot", font=fs, fill="white", anchor="mm",
               stroke_width=5, stroke_fill=INK)
        # character
        mq = round(min(1.0, mouth_env[f]) * 6) / 6
        hand = int(bp) % 2
        spark = phase < 0.45
        key = (mq, hand, spark)
        if key not in cache:
            cache[key] = draw_giggy(mq, hand, spark)
        h = 4 * phase * (1 - phase)
        sx, sy = 1.06 - 0.10 * h, 0.93 + 0.12 * h
        g = cache[key].resize((int(640 * sx), int(640 * sy)), Image.BILINEAR)
        g = g.rotate(5 * math.sin(math.pi * bp), resample=Image.BICUBIC, expand=True)
        shadow_w = int(300 - 60 * h)
        d.ellipse((VW // 2 - shadow_w // 2, 862, VW // 2 + shadow_w // 2, 892), fill=(62, 160, 90))
        img.paste(g, (VW // 2 - g.width // 2, int(868 - g.height + 38 - 46 * h)), g)
        # lyrics
        text, prog, key2 = None, 0.0, None
        if bp < INTRO:
            text, key2 = title, "title"
        elif bp >= total_beats - OUTRO:
            text, key2 = "Bye bye!", "bye"
        else:
            for i, (b0, nb, v) in enumerate(slots):
                if b0 <= bp < b0 + nb:
                    text, key2 = lines[i], i
                    vs = b0 * BEAT + 0.25 * BEAT
                    prog = min(1.0, max(0.0, (t - vs) / (len(v) / SR)))
                    break
        if text:
            lay = get_layout(key2, text)
            k = int(prog * len(lay) + 0.999) if prog > 0 else 0
            y0 = 985
            for j, (w, x, y) in enumerate(lay):
                col = (255, 224, 70) if j < k else (255, 255, 255)
                d.text((x, y0 + y), w, font=ft, fill=col, stroke_width=7, stroke_fill=INK)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")


def make_thumb(title, path="thumb.jpg"):
    tw, th = 1280, 720
    top, bot = random.choice(SKIES)
    img = Image.new("RGB", (tw, th))
    d = ImageDraw.Draw(img)
    for y in range(th):
        t = y / th
        d.line([(0, y), (tw, y)], fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    d.ellipse((-300, 560, 1500, 1100), fill=(98, 205, 118), outline=(60, 150, 85), width=8)
    g = draw_giggy(0.8, 0, True).resize((680, 680), Image.LANCZOS)
    img.paste(g, (20, 22), g)
    d.text((960, 70), "GigglePlot", font=ImageFont.truetype(FONT, 56), fill="white",
           anchor="mm", stroke_width=5, stroke_fill=INK)
    size = 96
    while True:
        font = ImageFont.truetype(FONT, size)
        ls = wrap(title, font, d, 540)
        if len(ls) * (size + 16) <= 460 or size <= 56:
            break
        size -= 6
    y = 380 - (len(ls) - 1) * (size + 16) // 2
    for line in ls:
        d.text((960, y), line, font=font, fill=(255, 235, 80), anchor="mm",
               stroke_width=8, stroke_fill=INK)
        y += size + 16
    img.save(path, quality=90)


# -------------------------------------------------------------------- upload
def upload(title, lines, desc, tags):
    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    yt = build("youtube", "v3", credentials=creds)
    base = ["kids songs", "nursery rhymes", "children's music", "Giggy", "GigglePlot"]
    all_tags, size = [], 0
    for tg in tags + base:
        if tg.lower() not in [x.lower() for x in all_tags] and size + len(tg) < 400:
            all_tags.append(tg)
            size += len(tg) + 1
    lyrics = "\n".join(lines)
    description = (f"{desc}\n\nSing along with Giggy!\n\n{lyrics}\n\n"
                   "#shorts #kidssongs #nurseryrhymes #giggy")
    body = {
        "snippet": {
            "title": title[:88] + " #shorts",
            "description": description[:4900],
            "tags": all_tags,
            "categoryId": "24",
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {"privacyStatus": os.environ.get("PRIVACY", "public").strip() or "public",
                   "selfDeclaredMadeForKids": True},
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
    title, lines, desc, tags = make_song(os.environ.get("SONG_TOPIC", "").strip() or None)
    print(title)
    print("\n".join(lines))
    make_video(title, lines)
    upload(title, lines, desc, tags)
