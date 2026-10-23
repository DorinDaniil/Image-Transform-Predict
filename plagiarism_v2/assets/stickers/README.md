# Sticker assets (optional)

Drop PNG / WebP files **with an alpha channel** here (subfolders are fine):
emoji sets, icons, logos, arrows, speech bubbles, "approved"/"sale" stamps, etc.
`StickerBank` picks them with probability `sticker_asset_prob` and otherwise
draws a procedural sticker.

Rules:

* never put images from the training set or from the negative-pair dataset
  here — a donor image could appear in the same batch as a negative for the
  edited image and the labels would contradict each other;
* opaque files are allowed: a near-uniform background (colour of the top-left
  pixel) is knocked out automatically most of the time;
* keep them reasonably small (≤ 1024 px), they are rescaled to 12–34 % of the
  shorter image side anyway.

Open sets that work well: OpenMoji (CC BY-SA 4.0), Twemoji (CC BY 4.0),
Noto Emoji (Apache 2.0), Tabler / Feather icons (MIT) rendered to PNG.
