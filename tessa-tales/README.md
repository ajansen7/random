# Tessa Tales ❄

Turns Tessa's kids-camera pictures into storybooks told in her own voice. Runs only on this Mac, for free.

```bash
uv run tessa.py
```

That opens **the shelf** at http://localhost:4747: one snow globe per story.
- **Tap a globe**: it shakes, and story time plays (one big ▶ first, because browsers need a tap before sound).
- **✏️ on a globe**: the grown-up page for that story.
- **Plug in the camera card**: Olaf bounces onto the shelf with 📷 and the number of new pictures. Tap him and he copies and sorts them, and the new day stories appear with ✨. ⏏ ejects the card. The card is never changed, and pictures already copied are skipped.

## How a story gets made

1. **Olaf sorts the card.** Junk (black, blank, pockets, ceilings) goes into Marshmallow's pile. Bursts stack up, even with black frames in between, and a lone deliberate click gets Elsa's snowflake.
2. **One story per day.** Days with fewer than 4 good pictures fold into "Little things". If the camera clock is unset, the whole card is one story.
3. **Tessa talks about her pictures** (🎤 Tell it!). One picture at a time, she taps the snowflake and says whatever she likes ("look like a dino neck!"). It stops by itself when she goes quiet. Whisper writes down her words on the Mac. Her talk belongs to the picture, so it follows it into any story.
4. **✨ Weave.** A small language model on the Mac reads what she said and picks the pictures, puts them in order, suggests a title and adds a few short narrator lines (📜). It never changes or invents her words. Bridges that repeat her or make things up are thrown away, and day changes get a true one ("The next morning."). Add an optional hint ("we brought her baby brother home") for better results.
5. **Story time.** Each page plays the narrator line (read aloud by the Mac), then her own recording.

On the story page, three stickers let you (or her) adjust: ❤️ in the book, ⭐ a big special page, 👎 to Marshmallow's pile (everywhere, ↩️ brings it back). Tap a picture to see it big, hear what she said 💬, and edit its narrator line 📜.

## Where things live

`~/TessaTales/`: `originals/` (copied off the card), `stories/<id>/story.json`, `talk.json` + `talk/` (her picture talk), `nope.json` (thumbs-down), `cache/`.

## Tuning (`config.toml`)

- `[sieve]`, `[prompts]`, `[stacks]`, `[moments]`: how Olaf sorts. Re-sort the latest card with `uv run tessa.py sieve`.
- `[voice]`: the Whisper model.
- `[weave]`: `engine = "local"` (free, default; `local_model` can be a bigger MLX model for richer bridges) or `"claude"` (better writing, needs Claude API billing).

The first run downloads the models once: MobileCLIP2 (sorting), Whisper large-v3-turbo (about 1.6 GB) and Qwen2.5 7B (about 4 GB, weaving).
