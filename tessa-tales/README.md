# Tessa Tales ❄

Turns Tessa's kids-camera pictures into storybooks told in her own voice. Runs only on this Mac.

## Make a book

Plug in the camera's SD card, then:

```bash
uv run tessa.py import
```

1. Every new picture is copied into `~/TessaTales/originals/`. The card is never changed, and pictures already imported are skipped.
2. **Olaf sorts them.** Black, blank and junk frames get buried in **Marshmallow's pile**. What's left gets stacked (bursts of the same thing become one stack, even with black frames in between), and a lone click with quiet before and after gets **Elsa's snowflake**.
3. The pictures are split into **moments**: ten minutes with nothing worth keeping starts a new one.
4. Olaf drafts a book, and the make-the-book page opens at http://localhost:4747.

On the page, three stickers do everything, so Tessa can do it too:
- ❤️ love it: the picture goes in the book
- ⭐ the best!: it goes in as a big, special page
- 👎 nope: it slides into Marshmallow's pile (↩️ brings it back)

Tap a sticker again to undo it. Tap a picture to see it big. ▶ plays a stack slowly through its different pictures, and 🎞️ makes it a flip-book page. Drag pages in the 📖 strip to reorder them. The book's name and the moment names are for grown-ups: tap them to rename. Everything saves on its own, to `~/TessaTales/stories/<import>/story.json`. 🔊 turns the chimes off.

## Tessa's turn

Press **🎤 Tell it!** on the book page. One page at a time (★ the cover, each picture, ♥ the end):
- She looks at the picture and tells you about it. Hold the **space bar** while she talks, or let her tap the giant ❄ to start and again to stop. The snowflake grows with her voice.
- When she stops, snow bursts out and she hears herself right away. 🔈 plays it again. Recording again replaces it, and older takes stay on disk.
- Whisper writes down her words a second or two later. ✏️ (for grown-ups) fixes a mishearing. Her recording is the real thing.
- The big orange arrows turn pages, and the ❄ dots at the top light up as pages get told.
- ▶ is **story time**: the whole book full screen, in her voice, with her words floating in.

Recordings live in `~/TessaTales/stories/<import>/audio/`, and her words in `voice.json` next to the book. Whisper (`large-v3-turbo` via mlx-whisper) runs on the Mac. The first recording downloads it, about 1.6 GB.

To come back to it later:

```bash
uv run tessa.py
```

## Tuning

All the knobs are in `config.toml`: the descriptions Olaf compares pictures against, the junk thresholds, stacking and moment gaps. Sort the latest card again without re-copying:

```bash
uv run tessa.py sieve
```

Books you've already made are kept. Pages remember their own pictures.

Olaf uses Apple's MobileCLIP2 model running locally on the Mac's GPU. Nothing leaves the Mac.
