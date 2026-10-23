"""
Local edits for plagiarism_v2 (PIL + numpy only; OpenCV used for inpainting
when it happens to be installed).

Two vocabulary tokens are produced here:

``text_overlay`` -- text put on (or next to) the image the way people do it.
    * content: real sentences from ``assets/text/<lang>.txt`` (~15 languages),
      an optional user corpus, and generic structured strings (dates, times,
      numbers, prices, URLs, e-mails, handles, phone numbers, copyright lines,
      localized figure / table captions). No hard-coded vocabulary.
    * scale: measured in what the network sees. Every string is sized so that
      its height after the 300x300 preprocessing is at least ``min_net_px`` and
      its proportions follow the layout (caption ~4-6 % of the image height,
      meme ~9-13 %, title ~10-18 %, watermark 3.5-6 %, ...).
    * background awareness: colour, outline, shadow and backing are chosen from
      the luminance / texture of the patch the text lands on; "free" placements
      prefer smooth areas, like a human looking for a readable spot.
    * layouts (fixed list ``TEXT_LAYOUTS``): corner_watermark,
      diagonal_watermark, caption_bar, caption_outside (white/black strip and
      caption added under / above the image like a figure in a document),
      subtitle, meme, label_tag, title, paragraph_block, timestamp, free.

``fragment_edit`` -- every OTHER local change (fixed list ``FRAGMENT_KINDS``):
    sticker          procedural stickers / badges / icons / QR / barcode, or PNGs
                     from a dedicated asset folder
    annotation       mark-up a person adds to point at something: outline box /
                     ellipse, arrow, underline / scribble, highlighter
    blur_censor      strong blur of a region (face / plate style censoring)
    pixelate_censor  mosaic over a region
    black_box        solid censor bar / patch
    copy_move        a patch of the same image cloned elsewhere (brightness matched)
    inpaint_fill     object removal: OpenCV inpainting if available, else a
                     stretched neighbouring band
    recolor          hue / saturation change of the dominant colour inside a
                     region (object-aware, not a flat colour blob)

Region masks are pixel masks of shape rect / ellipse / convex polygon / blob.

Design rules
    * no pixels are ever taken from other dataset images (a donor image could
      sit in the same BxB batch as a negative for the edited image);
    * a language is used only if an installed font has glyphs for it; scripts
      that need shaping (Arabic, Devanagari, ...) only when Pillow has libraqm.

All functions take and return RGB ``PIL.Image`` and use the global ``random``
module (same randomness source as the v1 ``ImageTransformer``).
"""

import glob
import math
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    from PIL import features as _pil_features
    RAQM_AVAILABLE = bool(_pil_features.check("raqm"))
except Exception:  # pragma: no cover
    RAQM_AVAILABLE = False

try:  # optional, only for inpaint_fill
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None
    CV2_AVAILABLE = False

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.normpath(os.path.join(_HERE, "..", "assets"))
TEXT_ASSETS_DIR = os.path.join(ASSETS_DIR, "text")

NET_SIDE = 300  # the model's input side after Resize((300, 300))


# =============================================================================
# Fonts
# =============================================================================

_DEFAULT_FONT_DIRS = [
    "C:/Windows/Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "/System/Library/Fonts",
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
]

# Symbol / icon fonts render garbage glyphs for plain text; skip by file-name hint.
_BAD_FONT_HINTS = (
    "symbol", "wingding", "webding", "marlett", "emoji", "bssym", "holomdl",
    "segmdl", "mtextra", "outlook", "refspcl", "mdl2", "dings", "icon", "mtproxy",
)

# Families people actually use for text on images (file-name stems, lower-case).
_COMMON_FAMILY_HINTS = (
    "arial", "helvetica", "times", "calibri", "cambria", "georgia", "verdana", "tahoma",
    "segoeui", "roboto", "opensans", "open_sans", "notosans", "noto_sans", "dejavusans",
    "liberationsans", "liberationserif", "ubuntu", "lato", "montserrat", "trebuc", "consola",
    "cour", "comic", "franklin", "garamond", "palatino", "century", "freesans", "freeserif",
    "ptsans", "ptserif", "sourcesans", "inter", "nunito", "malgun", "msyh", "yugoth", "meiryo",
    "simsun", "simhei", "nirmala", "mangal",
)
_MEME_FAMILY_HINTS = ("impact", "anton", "bebas", "oswald", "league")
_BOLD_HINTS = ("bold", "bd", "black", "heavy", "semibold", "sb", "extrab", "demib")
_ITALIC_HINTS = ("italic", "oblique")


def _render_glyph(font: ImageFont.ImageFont, ch: str) -> bytes:
    bbox = font.getbbox(ch)
    w, h = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((-bbox[0], -bbox[1]), ch, font=font, fill=255)
    return im.tobytes()


def font_renders(font: ImageFont.ImageFont, text: str) -> bool:
    """
    True when every non-space character of ``text`` has a real glyph in ``font``
    (i.e. is not drawn as the .notdef "tofu" box). The reference box is obtained
    by rendering a private-use code point that no text font defines.
    """
    try:
        notdef = _render_glyph(font, "\U0010FFFD")
        for ch in set(text):
            if ch.isspace():
                continue
            if _render_glyph(font, ch) == notdef:
                return False
        return True
    except Exception:
        return False


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].lower().replace("-", "").replace(" ", "").replace("_", "")


class FontBank:
    """
    Discovers every TrueType/OpenType font on the machine (plus optional extra
    directories), knows which fonts are "common", bold, italic or meme-style,
    and hands out fonts that actually have glyphs for a given text.
    """

    def __init__(self, font_dirs: Optional[Sequence[str]] = None, max_fonts: Optional[int] = None):
        dirs = list(font_dirs) if font_dirs else []
        dirs += _DEFAULT_FONT_DIRS
        paths: List[str] = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".ttf", ".otf")) and not any(h in fl for h in _BAD_FONT_HINTS):
                        paths.append(os.path.join(root, f))
        random.Random(0).shuffle(paths)
        self.paths = paths[:max_fonts] if max_fonts else paths
        self._bad = set()
        self._support_cache: Dict[Tuple[str, str], bool] = {}

        self.common = [p for p in self.paths if any(h in _stem(p) for h in _COMMON_FAMILY_HINTS)]
        self.meme = [p for p in self.paths if any(h in _stem(p) for h in _MEME_FAMILY_HINTS)]
        self.bold = [p for p in self.paths if any(h in _stem(p) for h in _BOLD_HINTS)]
        self.italic = [p for p in self.paths if any(h in _stem(p) for h in _ITALIC_HINTS)]

    @property
    def has_truetype(self) -> bool:
        return len(self.paths) > len(self._bad)

    @staticmethod
    def default(size: float) -> ImageFont.ImageFont:
        size = max(8, int(round(size)))
        try:
            return ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:  # very old Pillow: bitmap font, fixed size
            return ImageFont.load_default()

    def load(self, path: Optional[str], size: float) -> ImageFont.ImageFont:
        size = max(8, int(round(size)))
        if path is not None:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                self._bad.add(path)
        return self.default(size)

    def supports(self, path: str, probe: str) -> bool:
        key = (path, probe)
        if key not in self._support_cache:
            self._support_cache[key] = font_renders(self.load(path, 24), probe)
        return self._support_cache[key]

    def pool_for(self, probe: str) -> List[str]:
        """All font files that render every character of ``probe``."""
        return [p for p in self.paths if p not in self._bad and self.supports(p, probe)]

    def pick(self, pool: Sequence[str], text: str, size: float, style: Optional[str] = None,
             prefer_common: float = 0.7, tries: int = 10) -> Tuple[Optional[str], ImageFont.ImageFont]:
        """
        Random font from ``pool`` that renders ``text``; ``style`` in
        {None, "bold", "italic", "meme"} narrows the choice when such fonts exist.
        Common families are preferred with probability ``prefer_common``.
        Returns (path, font); (None, built-in) when nothing fits.
        """
        pool = [p for p in pool if p not in self._bad]
        if not pool:
            return None, self.default(size)
        styled = pool
        if style == "bold":
            styled = [p for p in pool if p in set(self.bold)] or pool
        elif style == "italic":
            styled = [p for p in pool if p in set(self.italic)] or pool
        elif style == "meme":
            styled = [p for p in pool if p in set(self.meme)] or [p for p in pool if p in set(self.bold)] or pool
        common = [p for p in styled if p in set(self.common)]
        for _ in range(tries):
            src = common if (common and random.random() < prefer_common) else styled
            p = random.choice(src)
            f = self.load(p, size)
            if p in self._bad:
                continue
            if font_renders(f, text):
                return p, f
        return None, self.default(size)


# =============================================================================
# Text bank: real phrases in many languages + generic structured generators
# =============================================================================

COMPLEX_SCRIPT_LANGS = {"ar", "he", "fa", "ur", "hi", "bn", "th"}   # need libraqm shaping
RTL_LANGS = {"ar", "he", "fa", "ur"}
NO_SPACE_LANGS = {"zh", "ja"}                                        # wrap by character
NO_CASE_LANGS = {"zh", "ja", "ko", "ar", "he", "fa", "ur", "hi", "bn", "th"}

DEFAULT_LANG_WEIGHTS = {
    "en": 4.0, "ru": 4.0, "de": 1.5, "fr": 1.5, "es": 1.5, "it": 1.0, "pt": 1.0,
    "pl": 1.0, "uk": 1.0, "tr": 1.0, "zh": 1.5, "ja": 1.0, "ko": 1.0, "ar": 0.7, "hi": 0.7,
    "corpus": 3.0,
}

# Localized *structural* labels of figures / tables / sources (not vocabulary).
CAPTION_TEMPLATES = {
    "en": ["Figure {n}", "Fig. {n}.", "Figure {n}: {w}", "Table {n}", "Source: {w}", "Photo: {w}", "Image {n}", "Fig. {n}. {w}"],
    "ru": ["Рис. {n}", "Рисунок {n}", "Рис. {n}. {w}", "Таблица {n}", "Источник: {w}", "Фото: {w}", "Рисунок {n} — {w}"],
    "de": ["Abbildung {n}", "Abb. {n}", "Abb. {n}: {w}", "Tabelle {n}", "Quelle: {w}", "Foto: {w}"],
    "fr": ["Figure {n}", "Fig. {n}", "Figure {n} : {w}", "Tableau {n}", "Source : {w}", "Photo : {w}"],
    "es": ["Figura {n}", "Fig. {n}", "Figura {n}: {w}", "Tabla {n}", "Fuente: {w}", "Foto: {w}"],
    "it": ["Figura {n}", "Fig. {n}", "Figura {n}: {w}", "Tabella {n}", "Fonte: {w}", "Foto: {w}"],
    "pt": ["Figura {n}", "Fig. {n}", "Figura {n}: {w}", "Tabela {n}", "Fonte: {w}", "Foto: {w}"],
    "pl": ["Rysunek {n}", "Rys. {n}", "Rys. {n}. {w}", "Tabela {n}", "Źródło: {w}", "Fot. {w}"],
    "uk": ["Рис. {n}", "Рисунок {n}", "Рис. {n}. {w}", "Таблиця {n}", "Джерело: {w}", "Фото: {w}"],
    "tr": ["Şekil {n}", "Şekil {n}: {w}", "Tablo {n}", "Kaynak: {w}", "Fotoğraf: {w}"],
    "zh": ["图 {n}", "图 {n}：{w}", "表 {n}", "来源：{w}", "图片：{w}"],
    "ja": ["図 {n}", "図 {n}：{w}", "表 {n}", "出典：{w}", "写真：{w}"],
    "ko": ["그림 {n}", "그림 {n}: {w}", "표 {n}", "출처: {w}", "사진: {w}"],
    "ar": ["الشكل {n}", "الشكل {n}: {w}", "الجدول {n}", "المصدر: {w}", "الصورة: {w}"],
    "hi": ["चित्र {n}", "चित्र {n}: {w}", "तालिका {n}", "स्रोत: {w}", "फ़ोटो: {w}"],
}

_TLDS = [".com", ".ru", ".org", ".net", ".io", ".de", ".fr", ".info", ".co", ".edu", ".me", ".eu"]
_CURRENCIES = ["$", "€", "£", "₽", "¥", "₴", "₺", "zł", "R$", "₹"]
_PUNCT_RE = re.compile(r"[\"'«»„“”‘’!?.,;:…()\[\]—–\-/]+")
_LATIN_RE = re.compile(r"[A-Za-z]")


class TextBank:
    """
    Real phrases per language (``assets/text/<lang>.txt``, one per line) plus an
    optional user corpus directory (any ``*.txt``, lines sampled, key ``corpus``),
    plus generic structured generators (dates, numbers, prices, URLs, e-mails,
    handles, phone numbers, copyright lines, localized captions).

    Only languages that an installed font can render are used; complex scripts
    additionally require libraqm (``RAQM_AVAILABLE``). ``lang_fonts[lang]`` is
    the pool of font files that render that language.
    """

    def __init__(
        self,
        font_bank: FontBank,
        corpus_dir: Optional[str] = None,
        langs: Optional[Sequence[str]] = None,
        lang_weights: Optional[Dict[str, float]] = None,
        min_corpus_line: int = 8,
        max_corpus_line: int = 160,
    ):
        self.font_bank = font_bank
        self.phrases: Dict[str, List[str]] = {}

        if os.path.isdir(TEXT_ASSETS_DIR):
            for p in sorted(glob.glob(os.path.join(TEXT_ASSETS_DIR, "*.txt"))):
                lang = os.path.splitext(os.path.basename(p))[0].lower()
                lines = self._read_lines(p, 1, 400)
                if lines:
                    self.phrases[lang] = lines

        if corpus_dir and os.path.isdir(corpus_dir):
            lines: List[str] = []
            for p in glob.glob(os.path.join(corpus_dir, "**", "*.txt"), recursive=True):
                lines += self._read_lines(p, min_corpus_line, max_corpus_line)
            if lines:
                random.Random(0).shuffle(lines)
                self.phrases["corpus"] = lines[:200000]

        if langs:
            keep = set(l.lower() for l in langs) | {"corpus"}
            self.phrases = {k: v for k, v in self.phrases.items() if k in keep}

        weights = dict(DEFAULT_LANG_WEIGHTS)
        if lang_weights:
            weights.update({str(k): float(v) for k, v in lang_weights.items()})

        # keep only languages that (a) have fonts with glyphs and (b) can be shaped
        self.lang_weights: Dict[str, float] = {}
        self.lang_fonts: Dict[str, List[str]] = {}
        for lang, lines in self.phrases.items():
            if lang in COMPLEX_SCRIPT_LANGS and not RAQM_AVAILABLE:
                continue
            probe = self._probe(lines)
            pool = self.font_bank.pool_for(probe)
            if not pool and not font_renders(self.font_bank.default(24), probe):
                continue
            self.lang_fonts[lang] = pool
            self.lang_weights[lang] = weights.get(lang, 1.0)
        if not self.lang_weights:  # last resort: built-in font, ASCII only
            self.phrases["en"] = self.phrases.get("en") or ["Sample text", "All rights reserved."]
            self.lang_weights["en"] = 1.0
            self.lang_fonts["en"] = []

    @staticmethod
    def _probe(lines: List[str], max_chars: int = 40) -> str:
        """Distinct non-space characters from the first lines, as a glyph probe."""
        seen = []
        for ln in lines[:12]:
            for ch in ln:
                if not ch.isspace() and ch not in seen and not ch.isdigit():
                    seen.append(ch)
                if len(seen) >= max_chars:
                    break
            if len(seen) >= max_chars:
                break
        return "".join(seen)

    @staticmethod
    def _read_lines(path: str, min_len: int, max_len: int) -> List[str]:
        out = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().split())
                    if min_len <= len(s) <= max_len and not s.startswith("#"):
                        out.append(s)
        except OSError:
            pass
        return out

    # ----------------------------------------------------------- sampling
    @property
    def langs(self) -> List[str]:
        return sorted(self.lang_weights)

    def pick_lang(self) -> str:
        langs = list(self.lang_weights)
        return random.choices(langs, weights=[self.lang_weights[l] for l in langs])[0]

    def fonts(self, lang: str) -> List[str]:
        return self.lang_fonts.get(lang) or self.lang_fonts.get("en") or self.font_bank.paths

    def phrase(self, lang: str) -> str:
        return random.choice(self.phrases[lang])

    def short(self, lang: str, max_units: int = 3) -> str:
        """1..max_units words (or 2..6 characters for zh/ja) taken from a real phrase."""
        s = _PUNCT_RE.sub(" ", self.phrase(lang)).strip()
        if lang in NO_SPACE_LANGS:
            s = s.replace(" ", "")
            n = random.randint(2, min(6, max(2, len(s))))
            start = random.randint(0, max(0, len(s) - n))
            return s[start:start + n]
        words = [w for w in s.split() if w]
        if not words:
            return s
        n = random.randint(1, min(max_units, len(words)))
        start = random.randint(0, len(words) - n)
        return " ".join(words[start:start + n])

    def caption(self, lang: str) -> str:
        templates = CAPTION_TEMPLATES.get(lang) or CAPTION_TEMPLATES["en"]
        t = random.choice(templates)
        return t.format(n=random.randint(1, 14), w=self.short(lang, 3))

    def brand(self, lang: str) -> str:
        """A name-like token: a word from the phrase bank or a pseudo-word."""
        if random.random() < 0.6:
            w = self.short(lang, 1)
            return w.capitalize() if lang not in NO_CASE_LANGS else w
        return self.pseudo_word().capitalize()

    @staticmethod
    def pseudo_word(n: Optional[int] = None) -> str:
        """Pronounceable Latin pseudo-word for domains / handles (no fixed vocabulary)."""
        cons, vow = "bcdfghklmnprstvz", "aeiou"
        n = n or random.randint(4, 8)
        out = []
        for i in range(n):
            out.append(random.choice(cons) if i % 2 == 0 else random.choice(vow))
        return "".join(out)

    def watermark(self, lang: str) -> str:
        r = random.random()
        if r < 0.35:
            return self.copyright(lang)
        if r < 0.6:
            return self.url()
        if r < 0.75:
            return self.handle()
        s = self.short(lang, 2)
        return s.upper() if lang not in NO_CASE_LANGS else s

    def copyright(self, lang: str) -> str:
        year = random.randint(2004, 2026)
        brand = self.brand(lang)
        return random.choice([f"© {year} {brand}", f"© {brand}", f"(c) {year} {brand}", f"© {year}", f"{brand} © {year}"])

    def url(self) -> str:
        host = self.pseudo_word() + random.choice(["", str(random.randint(1, 99)), "-" + self.pseudo_word(4)]) + random.choice(_TLDS)
        return random.choice([f"www.{host}", f"https://{host}", host, f"{host}/{self.pseudo_word(5)}"])

    def email(self) -> str:
        return f"{self.pseudo_word()}{random.choice(['', '.', '_'])}{random.choice(['', self.pseudo_word(4), str(random.randint(1, 99))])}@{self.pseudo_word()}{random.choice(_TLDS)}"

    def handle(self) -> str:
        return "@" + self.pseudo_word() + random.choice(["", str(random.randint(1, 99)), "_" + self.pseudo_word(3)])

    @staticmethod
    def phone() -> str:
        return random.choice([
            f"+{random.randint(1, 99)} {random.randint(900, 999)} {random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(10, 99)}",
            f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            f"8 {random.randint(800, 999)} {random.randint(100, 999)} {random.randint(10, 99)} {random.randint(10, 99)}",
        ])

    @staticmethod
    def timestamp() -> str:
        y, mo, d = random.randint(2005, 2026), random.randint(1, 12), random.randint(1, 28)
        hh, mm = random.randint(0, 23), random.randint(0, 59)
        return random.choice([
            f"{y}/{mo:02d}/{d:02d} {hh:02d}:{mm:02d}", f"{d:02d}.{mo:02d}.{y}", f"{y}-{mo:02d}-{d:02d}",
            f"{d:02d}/{mo:02d}/{y} {hh:02d}:{mm:02d}", f"{mo:02d}/{d:02d}/{y}", f"{hh:02d}:{mm:02d}",
            f"{d:02d}.{mo:02d}.{y} {hh:02d}:{mm:02d}:{random.randint(0, 59):02d}",
        ])

    @staticmethod
    def number() -> str:
        r = random.random()
        if r < 0.3:
            return str(random.randint(0, 9999))
        if r < 0.5:
            return f"{random.randint(1, 100)}%"
        if r < 0.65:
            return f"#{random.randint(1, 9999)}"
        if r < 0.8:
            return f"{random.randint(1, 48)} / {random.randint(49, 120)}"
        return f"{random.uniform(0, 100):.{random.randint(1, 2)}f}"

    @staticmethod
    def price() -> str:
        cur = random.choice(_CURRENCIES)
        val = random.choice([random.randint(1, 99), random.randint(100, 9999), random.randint(10, 999) * 10])
        s = f"{val:,}".replace(",", " ") if val >= 10000 else str(val)
        return random.choice([f"{cur}{s}", f"{s} {cur}", f"{s}{cur}", f"{s}.{random.randint(0, 99):02d} {cur}"])

    def label(self, lang: str) -> str:
        """Short tag-like string: a couple of words, a number, a price, a date, a handle."""
        r = random.random()
        if r < 0.45:
            return self.short(lang, 2)
        if r < 0.6:
            return self.price()
        if r < 0.75:
            return self.number()
        if r < 0.9:
            return self.timestamp()
        return self.handle()

    def any_text(self, lang: str) -> str:
        r = random.random()
        if r < 0.45:
            return self.phrase(lang)
        if r < 0.65:
            return self.short(lang, 4)
        if r < 0.8:
            return self.caption(lang)
        return random.choice([self.url(), self.email(), self.phone(), self.timestamp(), self.copyright(lang)])


# =============================================================================
# Colours and text rendering helpers
# =============================================================================

_PALETTE = [
    (230, 40, 40), (255, 140, 0), (255, 215, 0), (40, 170, 70), (30, 120, 220),
    (140, 60, 200), (255, 100, 160), (20, 20, 20), (250, 250, 250), (0, 180, 180),
]
_WHITE, _BLACK, _YELLOW = (255, 255, 255), (0, 0, 0), (255, 220, 0)


def _luma(c: Tuple[int, int, int]) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _contrast(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return _WHITE if _luma(c) < 128 else _BLACK


def _font_px(font: ImageFont.ImageFont, fallback: float) -> int:
    return int(getattr(font, "size", fallback) or fallback)


def _text_bbox(font, text, stroke_w=0, direction=None):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    kw = {"stroke_width": stroke_w}
    if direction:
        kw["direction"] = direction
    try:
        return d.textbbox((0, 0), text, font=font, **kw)
    except Exception:  # raqm missing / direction unsupported
        return d.textbbox((0, 0), text, font=font, stroke_width=stroke_w)


def wrap_text(text: str, font: ImageFont.ImageFont, max_w: float, by_char: bool, max_lines: int,
              direction: Optional[str] = None) -> List[str]:
    """Greedy word (or character) wrap; truncates with an ellipsis past ``max_lines``."""
    units = list(text.replace(" ", "")) if by_char else text.split()
    joiner = "" if by_char else " "
    lines: List[str] = []
    cur = ""
    for u in units:
        cand = (cur + joiner + u) if cur else u
        bb = _text_bbox(font, cand, 0, direction)
        if bb[2] - bb[0] <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = u
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    elif cur and lines:
        lines[-1] = lines[-1].rstrip(" ,;:") + "…"
    return lines or [text]


def render_text_block(
    lines: List[str],
    font: ImageFont.ImageFont,
    fill: Optional[Tuple[int, int, int]],
    stroke_w: int = 0,
    stroke_fill: Optional[Tuple[int, int, int]] = None,
    align: str = "left",
    spacing: float = 0.25,
    direction: Optional[str] = None,
    shadow: Optional[Tuple[int, float, float]] = None,
) -> Image.Image:
    """
    Tight RGBA layer with the given lines.
    ``fill=None`` draws outline-only text (needs stroke_w > 0).
    ``shadow=(offset_px, blur_px, alpha)`` adds a soft dark drop shadow.
    """
    px = _font_px(font, 20)
    line_h = int(px * (1.0 + spacing))
    bbs = [_text_bbox(font, ln, stroke_w, direction) for ln in lines]
    widths = [bb[2] - bb[0] for bb in bbs]
    pad = 0
    if shadow:
        pad = int(shadow[0] + shadow[1] * 2 + 1)
    W = max(1, max(widths) + 2 * stroke_w + 2 * pad)
    H = max(1, line_h * len(lines) + 2 * stroke_w + 2 * pad)

    def draw_lines(layer: Image.Image, color_rgba, stroke_rgba, dx=0, dy=0) -> None:
        d = ImageDraw.Draw(layer)
        y = stroke_w + pad + dy
        for ln, bb, lw in zip(lines, bbs, widths):
            if align == "center":
                x = (W - lw) // 2
            elif align == "right":
                x = W - lw - stroke_w - pad
            else:
                x = stroke_w + pad
            x += dx
            kw = {"font": font, "fill": color_rgba}
            if stroke_w:
                kw.update(stroke_width=stroke_w, stroke_fill=stroke_rgba)
            if direction:
                kw["direction"] = direction
            try:
                d.text((x - bb[0], y - bb[1]), ln, **kw)
            except Exception:
                kw.pop("direction", None)
                d.text((x - bb[0], y - bb[1]), ln, **kw)
            y += line_h

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if shadow:
        off, blur, alpha = shadow
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_lines(sh, (0, 0, 0, 255), (0, 0, 0, 255), off, off)
        if blur > 0.3:
            sh = sh.filter(ImageFilter.GaussianBlur(blur))
        sh = _scale_alpha(sh, alpha)
        out.alpha_composite(sh)
    text_rgba = (fill + (255,)) if fill is not None else (0, 0, 0, 0)
    stroke_rgba = ((stroke_fill if stroke_fill is not None else (_contrast(fill) if fill else _WHITE)) + (255,))
    draw_lines(out, text_rgba, stroke_rgba)
    return out


def _scale_alpha(layer: Image.Image, factor: float) -> Image.Image:
    if factor >= 0.999:
        return layer
    r, g, b, a = layer.split()
    a = a.point(lambda v: int(v * factor))
    return Image.merge("RGBA", (r, g, b, a))


def _fit_layer(layer: Image.Image, w: int, h: int, max_frac: float = 0.95) -> Image.Image:
    if layer.width <= w * max_frac and layer.height <= h * max_frac:
        return layer
    s = min(w * max_frac / layer.width, h * max_frac / layer.height)
    nw, nh = max(1, int(layer.width * s)), max(1, int(layer.height * s))
    return layer.resize((nw, nh), Image.LANCZOS)


def _composite_at(base: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """Alpha-composite ``layer`` onto RGBA ``base`` at (x, y), clipping to the canvas."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(base.width, x + layer.width), min(base.height, y + layer.height)
    if x1 <= x0 or y1 <= y0:
        return
    crop = layer.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    base.alpha_composite(crop, (x0, y0))


def _soft_shadow(layer: Image.Image, radius: float, alpha: float = 0.5) -> Image.Image:
    a = layer.split()[-1].filter(ImageFilter.GaussianBlur(max(0.5, radius)))
    a = a.point(lambda v: int(v * alpha))
    return Image.merge("RGBA", (Image.new("L", layer.size, 0),) * 3 + (a,))


def _patch_stats(img: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """(mean luma, luma std) of the RGB image inside box=(x, y, w, h), clipped."""
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.width, x + w), min(img.height, y + h)
    if x1 <= x0 or y1 <= y0:
        return 128.0, 0.0
    region = img.crop((x0, y0, x1, y1)).convert("L")
    if region.width * region.height > 64 * 64:
        region = region.resize((64, max(1, int(64 * region.height / region.width))), Image.BILINEAR)
    arr = np.asarray(region, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def _text_color_for(bg_luma: float, allow_colour: float = 0.2) -> Tuple[int, int, int]:
    """Readable text colour for a patch of given mean luminance."""
    if random.random() < allow_colour:
        cands = [c for c in _PALETTE if abs(_luma(c) - bg_luma) >= 70]
        if cands:
            return random.choice(cands)
    if bg_luma < 100:
        return _WHITE if random.random() < 0.9 else _YELLOW
    if bg_luma > 160:
        return _BLACK if random.random() < 0.85 else random.choice([(40, 40, 120), (120, 20, 20)])
    return _WHITE if random.random() < 0.6 else _BLACK


def _with_background(layer: Image.Image, bg: Tuple[int, int, int], alpha: int, pad: int,
                     radius: int = 0, full_width: Optional[int] = None, align: str = "left",
                     gradient: bool = False) -> Image.Image:
    """
    Put ``layer`` on a (rounded) rectangle; ``full_width`` makes a bar spanning the
    image; ``gradient`` fades the bar from transparent (top) to ``alpha`` (bottom).
    """
    W = full_width if full_width else layer.width + 2 * pad
    H = layer.height + 2 * pad
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if gradient:
        col = np.zeros((H, W, 4), np.uint8)
        col[..., 0], col[..., 1], col[..., 2] = bg
        col[..., 3] = (np.linspace(0, alpha, H)[:, None] * np.ones((1, W))).astype(np.uint8)
        out = Image.fromarray(col, "RGBA")
    else:
        d = ImageDraw.Draw(out)
        if radius > 0:
            d.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius, fill=bg + (alpha,))
        else:
            d.rectangle([0, 0, W - 1, H - 1], fill=bg + (alpha,))
    if full_width:
        x = pad if align == "left" else ((W - layer.width) // 2 if align == "center" else W - pad - layer.width)
    else:
        x = pad
    out.alpha_composite(layer, (max(0, x), pad))
    return out


# =============================================================================
# Text layouts (how people actually put text on images)
# =============================================================================

TEXT_LAYOUTS: Dict[str, float] = {
    "corner_watermark": 2.0,
    "diagonal_watermark": 1.5,
    "caption_bar": 2.0,
    "caption_outside": 2.0,
    "subtitle": 2.0,
    "meme": 1.5,
    "label_tag": 1.5,
    "title": 1.5,
    "paragraph_block": 1.0,
    "timestamp": 1.0,
    "free": 1.5,
}

# Text height as a fraction of the reference side sqrt(w*h): (min, max) per layout.
_LAYOUT_SIZE = {
    "corner_watermark": (0.035, 0.065),
    "diagonal_watermark": (0.10, 0.22),
    "caption_bar": (0.045, 0.075),
    "caption_outside": (0.040, 0.070),
    "subtitle": (0.055, 0.090),
    "meme": (0.085, 0.130),
    "label_tag": (0.055, 0.095),
    "title": (0.100, 0.180),
    "paragraph_block": (0.040, 0.055),
    "timestamp": (0.045, 0.070),
    "free": (0.055, 0.120),
}
_GLYPH_PER_EM = 0.72  # typical cap height / em size: used to turn a glyph-height requirement into a font size


class TextOverlayer:
    """
    Puts text on an image in one of the fixed layouts, sized for the network's
    300x300 view and coloured / backed according to the patch it lands on.
    """

    def __init__(self, font_bank: FontBank, text_bank: TextBank, layout_weights: Optional[Dict[str, float]] = None,
                 min_net_px: float = 9.0):
        self.fonts = font_bank
        self.texts = text_bank
        self.layout_weights = dict(TEXT_LAYOUTS)
        if layout_weights:
            self.layout_weights.update({str(k): float(v) for k, v in layout_weights.items()})
        self.min_net_px = float(min_net_px)
        self.last_layout: Optional[str] = None
        self.last_lang: Optional[str] = None

    # -------------------------------------------------------------- sizing
    def _size(self, w: int, h: int, layout: str, lines: int = 1) -> float:
        """Font size in pixels: layout fraction of sqrt(w*h), raised so that the text
        is at least ``min_net_px`` tall after the 300x300 preprocessing."""
        ref = math.sqrt(w * h)
        lo, hi = _LAYOUT_SIZE[layout]
        size = ref * random.uniform(lo, hi)
        # glyphs are ~0.72 em tall and the height axis is squashed by h/300
        min_px = self.min_net_px / _GLYPH_PER_EM * h / NET_SIDE
        size = max(size, min_px)
        size = min(size, 0.5 * h / max(1, lines), 0.9 * w)
        return max(8.0, size)

    # -------------------------------------------------------------- helpers
    def _block(self, text: str, lang: str, size: float, color: Optional[Tuple[int, int, int]], stroke_w: int = 0,
               stroke_color=None, max_w: Optional[float] = None, max_lines: int = 2, align: str = "left",
               uppercase: bool = False, style: Optional[str] = None, shadow=None, spacing: float = 0.25) -> Image.Image:
        if uppercase and lang not in NO_CASE_LANGS:
            text = text.upper()
        direction = "rtl" if (lang in RTL_LANGS and RAQM_AVAILABLE) else None
        path, font = self.fonts.pick(self.texts.fonts(lang), text, size, style=style)
        if path is None and not font_renders(font, text):  # built-in font cannot draw this script
            text = self.texts.phrase("en") if "en" in self.texts.phrases else "Sample text"
            lang, direction = "en", None
            if uppercase:
                text = text.upper()
        by_char = lang in NO_SPACE_LANGS
        lines = wrap_text(text, font, max_w if max_w else 10 ** 6, by_char, max_lines, direction)
        if max_w:  # shrink if a single unit is still wider than max_w
            for _ in range(4):
                widest = max(_text_bbox(font, ln, stroke_w, direction)[2] for ln in lines)
                if widest <= max_w or size <= 8:
                    break
                size *= 0.85 * max_w / widest
                font = self.fonts.load(path, size)
                lines = wrap_text(text, font, max_w, by_char, max_lines, direction)
        return render_text_block(lines, font, color, stroke_w, stroke_color, align, spacing, direction, shadow)

    @staticmethod
    def _corner_pos(w: int, h: int, lw: int, lh: int, where: str, margin: int) -> Tuple[int, int]:
        return {
            "tl": (margin, margin), "tr": (w - lw - margin, margin),
            "bl": (margin, h - lh - margin), "br": (w - lw - margin, h - lh - margin),
            "tc": ((w - lw) // 2, margin), "bc": ((w - lw) // 2, h - lh - margin),
            "c": ((w - lw) // 2, (h - lh) // 2),
        }[where]

    @staticmethod
    def _smooth_spot(img: Image.Image, lw: int, lh: int, margin: int, k: int = 4) -> Tuple[int, int]:
        """Of k random positions pick the one with the least texture (where a human
        would put text so it stays readable)."""
        w, h = img.size
        best, best_std = None, None
        for _ in range(k):
            x = random.randint(margin, max(margin, w - lw - margin))
            y = random.randint(margin, max(margin, h - lh - margin))
            _, std = _patch_stats(img, (x, y, lw, lh))
            if best is None or std < best_std:
                best, best_std = (x, y), std
        return best

    def _decorate(self, img: Image.Image, text: str, lang: str, size: float, box_probe: Tuple[int, int, int, int],
                  max_w: float, max_lines: int, align: str, uppercase: bool = False, style: Optional[str] = None,
                  allow_colour: float = 0.2, force_backing: bool = False) -> Image.Image:
        """
        Render ``text`` so that it is readable over the patch ``box_probe``:
        colour from the patch luminance; on textured patches add an outline, a
        shadow or a translucent backing, like people do.
        """
        luma, std = _patch_stats(img, box_probe)
        color = _text_color_for(luma, allow_colour)
        busy = std > 28 or abs(_luma(color) - luma) < 60
        r = random.random()
        stroke_w, shadow, backing = 0, None, None
        if force_backing or (busy and r < 0.3):
            backing = "box"
        elif busy and r < 0.65:
            stroke_w = max(1, int(size / random.uniform(10, 18)))
        elif busy or r < 0.3:
            shadow = (max(1, int(size * 0.05)), size * 0.06, random.uniform(0.5, 0.8))
        layer = self._block(text, lang, size, color, stroke_w=stroke_w, stroke_color=_contrast(color),
                            max_w=max_w, max_lines=max_lines, align=align, uppercase=uppercase, style=style, shadow=shadow)
        if backing == "box":
            bg = _contrast(color) if random.random() < 0.8 else random.choice(_PALETTE)
            if _luma(bg) > 128 and color == _WHITE:
                color = _BLACK
            layer = self._block(text, lang, size, _contrast(bg), max_w=max_w, max_lines=max_lines, align=align,
                                uppercase=uppercase, style=style)
            layer = _with_background(layer, bg, random.randint(120, 230), int(size * 0.4), radius=int(size * random.choice([0, 0.2, 0.5])))
        return layer

    # -------------------------------------------------------------- layouts
    def apply(self, image: Image.Image, layout: Optional[str] = None) -> Image.Image:
        img = image.convert("RGB")
        w, h = img.size
        ref = math.sqrt(w * h)
        lang = self.texts.pick_lang()
        if layout is None:
            names = list(self.layout_weights)
            layout = random.choices(names, weights=[self.layout_weights[n] for n in names])[0]
        self.last_layout, self.last_lang = layout, lang
        T = self.texts
        margin = int(0.025 * ref)

        if layout == "caption_outside":
            return self._caption_outside(img, lang)

        base = img.convert("RGBA")

        if layout == "corner_watermark":
            text = T.watermark(lang)
            size = self._size(w, h, layout)
            where = random.choice(["br", "br", "br", "bl", "tr", "tl"])
            probe_w, probe_h = int(size * 0.55 * max(4, len(text))), int(size * 1.3)
            px, py = self._corner_pos(w, h, probe_w, probe_h, where, margin)
            luma, std = _patch_stats(img, (px, py, probe_w, probe_h))
            color = _WHITE if luma < 150 else _BLACK
            if random.random() < 0.15:
                color = random.choice(_PALETTE)
            stroke_w = max(1, int(size / 16)) if (std > 25 or random.random() < 0.3) else 0
            shadow = None if stroke_w else (1, size * 0.05, 0.6)
            layer = self._block(text, lang, size, color, stroke_w=stroke_w, stroke_color=_contrast(color),
                                max_w=0.6 * w, max_lines=1, shadow=shadow)
            layer = _scale_alpha(layer, random.uniform(0.55, 1.0))
            layer = _fit_layer(layer, w, h, 0.6)
            x, y = self._corner_pos(w, h, layer.width, layer.height, where, margin)
            _composite_at(base, layer, x, y)

        elif layout == "diagonal_watermark":
            text = T.watermark(lang)
            size = self._size(w, h, layout)
            luma, _ = _patch_stats(img, (0, 0, w, h))
            color = random.choice([_WHITE, _WHITE, _BLACK, (128, 128, 128), (230, 40, 40)])
            if abs(_luma(color) - luma) < 50:
                color = _contrast((int(luma),) * 3)
            outline_only = random.random() < 0.3
            layer = self._block(text, lang, size, None if outline_only else color,
                                stroke_w=max(1, int(size / 14)) if outline_only else 0, stroke_color=color,
                                max_w=1.3 * w, max_lines=1, uppercase=random.random() < 0.5)
            layer = _scale_alpha(layer, random.uniform(0.25, 0.6))
            angle = random.choice([random.uniform(-45, -25), random.uniform(25, 45), 0.0])
            if abs(angle) > 0.5:
                layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
            layer = _fit_layer(layer, w, h, 0.95)
            lw, lh = layer.size
            positions = [((w - lw) // 2, (h - lh) // 2)]
            if random.random() < 0.4:  # tiled
                step_x, step_y = max(lw + margin, w // 2), max(lh + margin, h // 3)
                positions = [(x, y) for y in range(-lh // 2, h, step_y) for x in range(-lw // 2, w, step_x)]
            for (x, y) in positions:
                _composite_at(base, layer, x, y)

        elif layout == "caption_bar":
            text = T.phrase(lang) if random.random() < 0.6 else T.caption(lang)
            size = self._size(w, h, layout)
            at_bottom = random.random() < 0.8
            r = random.random()
            if r < 0.45:   # solid / translucent dark or light bar
                bg = random.choice([_BLACK, _BLACK, (30, 30, 30), _WHITE, (20, 40, 80), (120, 20, 20)])
                alpha, gradient = random.randint(150, 235), False
            elif r < 0.8:  # gradient fade like photo captions
                bg, alpha, gradient = _BLACK, random.randint(170, 230), at_bottom
                if not at_bottom:
                    gradient = False
            else:          # frosted: blurred & darkened strip of the image itself
                bg, alpha, gradient = None, 0, False
            align = "left" if random.random() < 0.6 else "center"
            color = _contrast(bg) if bg else _WHITE
            layer = self._block(text, lang, size, color, max_w=0.9 * w, max_lines=2, align=align,
                                stroke_w=0 if bg else max(1, int(size / 16)))
            pad = int(size * 0.5)
            if bg is None:
                bar_h = layer.height + 2 * pad
                y = h - bar_h if at_bottom else 0
                strip = img.crop((0, y, w, y + bar_h)).filter(ImageFilter.GaussianBlur(size * 0.4))
                strip = Image.fromarray((np.asarray(strip).astype(np.float32) * 0.55).astype(np.uint8))
                base.paste(strip, (0, y))
                x = pad if align == "left" else (w - layer.width) // 2
                _composite_at(base, layer, x, y + pad)
            else:
                bar = _with_background(layer, bg, alpha, pad, 0, full_width=w, align=align, gradient=gradient)
                if gradient:  # gradient bars are taller: fade starts above the text
                    extra = int(bar.height * random.uniform(0.6, 1.2))
                    col = np.zeros((extra + bar.height, w, 4), np.uint8)
                    col[..., 3] = (np.linspace(0, alpha, extra + bar.height)[:, None] * np.ones((1, w))).astype(np.uint8)
                    big = Image.fromarray(col, "RGBA")
                    big.alpha_composite(_with_background(layer, bg, 0, pad, 0, full_width=w, align=align), (0, extra))
                    bar = big
                y = h - bar.height if at_bottom else 0
                _composite_at(base, bar, 0, y)

        elif layout == "subtitle":
            text = T.phrase(lang)
            size = self._size(w, h, layout, lines=2)
            color = _WHITE if random.random() < 0.8 else _YELLOW
            layer = self._block(text, lang, size, color, stroke_w=max(1, int(size / 14)), stroke_color=_BLACK,
                                max_w=0.9 * w, max_lines=2, align="center",
                                shadow=(1, size * 0.04, 0.5) if random.random() < 0.5 else None)
            x, y = self._corner_pos(w, h, layer.width, layer.height, "bc", int(random.uniform(0.04, 0.09) * h))
            _composite_at(base, layer, x, y)

        elif layout == "meme":
            size = self._size(w, h, layout, lines=2)
            slots = random.choice([["tc"], ["bc"], ["tc", "bc"], ["tc", "bc"]])
            for slot in slots:
                text = T.phrase(lang) if random.random() < 0.7 else T.short(lang, 4)
                layer = self._block(text, lang, size, _WHITE, stroke_w=max(2, int(size / 10)), stroke_color=_BLACK,
                                    max_w=0.94 * w, max_lines=2, align="center", uppercase=True, style="meme", spacing=0.1)
                x, y = self._corner_pos(w, h, layer.width, layer.height, slot, int(0.03 * h))
                _composite_at(base, layer, x, y)

        elif layout == "label_tag":
            text = T.label(lang)
            size = self._size(w, h, layout)
            bg = random.choice(_PALETTE + [_WHITE, _WHITE, _BLACK, _BLACK])
            layer = self._block(text, lang, size, _contrast(bg), max_w=0.6 * w, max_lines=1,
                                style="bold" if random.random() < 0.5 else None)
            pad = int(size * 0.35)
            tag = _with_background(layer, bg, random.randint(215, 255), pad, radius=int(pad * random.choice([0, 0.6, 1.5])))
            if random.random() < 0.3:
                tag = tag.rotate(random.uniform(-12, 12), expand=True, resample=Image.BICUBIC)
            tag = _fit_layer(tag, w, h, 0.8)
            if random.random() < 0.5:  # near a corner, like a badge
                where = random.choice(["tl", "tr", "bl", "br"])
                x, y = self._corner_pos(w, h, tag.width, tag.height, where, margin)
            else:
                x = random.randint(margin, max(margin, w - tag.width - margin))
                y = random.randint(margin, max(margin, h - tag.height - margin))
            if random.random() < 0.5:
                off = max(1, int(size * 0.06))
                _composite_at(base, _soft_shadow(tag, size * 0.08, 0.45), x + off, y + off)
            _composite_at(base, tag, x, y)

        elif layout == "title":
            text = T.short(lang, 4) if random.random() < 0.6 else T.caption(lang)
            size = self._size(w, h, layout, lines=2)
            probe_w, probe_h = int(min(0.9 * w, size * 0.55 * max(4, len(text)))), int(size * 1.4)
            if random.random() < 0.6:
                x, y = self._smooth_spot(img, probe_w, probe_h, margin)
                align = "left"
            else:
                where = random.choice(["tc", "c", "tl", "bl"])
                x, y = self._corner_pos(w, h, probe_w, probe_h, where, int(0.05 * ref))
                align = "center" if where in ("tc", "c") else "left"
            layer = self._decorate(img, text, lang, size, (x, y, probe_w, probe_h), 0.9 * w, 2, align,
                                   style="bold" if random.random() < 0.6 else None, allow_colour=0.3)
            layer = _fit_layer(layer, w, h, 0.95)
            x = min(max(0, x), w - layer.width)
            y = min(max(0, y), h - layer.height)
            _composite_at(base, layer, x, y)

        elif layout == "paragraph_block":
            n = random.randint(2, 4)
            text = " ".join(T.phrase(lang) for _ in range(n))
            size = self._size(w, h, layout, lines=4)
            block_w = random.uniform(0.4, 0.85) * w
            probe_h = int(size * 1.25 * 4)
            x, y = self._smooth_spot(img, int(block_w), probe_h, margin)
            layer = self._decorate(img, text, lang, size, (x, y, int(block_w), probe_h), block_w, 6,
                                   random.choice(["left", "left", "center"]), allow_colour=0.1,
                                   force_backing=random.random() < 0.5)
            layer = _fit_layer(layer, w, h, 0.95)
            x = min(max(0, x), w - layer.width)
            y = min(max(0, y), h - layer.height)
            _composite_at(base, layer, x, y)

        elif layout == "timestamp":
            text = T.timestamp()
            size = self._size(w, h, layout)
            color = random.choice([(255, 170, 0), (255, 170, 0), _YELLOW, _WHITE])
            layer = self._block(text, "en", size, color, stroke_w=1, stroke_color=(60, 30, 0), max_w=0.6 * w, max_lines=1,
                                shadow=(1, size * 0.05, 0.5))
            layer = _fit_layer(layer, w, h, 0.6)
            where = random.choice(["br", "br", "bl", "tr"])
            x, y = self._corner_pos(w, h, layer.width, layer.height, where, int(0.03 * ref))
            _composite_at(base, layer, x, y)

        else:  # free
            text = T.any_text(lang)
            size = self._size(w, h, layout, lines=2)
            probe_w, probe_h = int(min(0.9 * w, size * 0.55 * max(4, len(text)))), int(size * 1.4)
            if random.random() < 0.7:
                x, y = self._smooth_spot(img, probe_w, probe_h, margin, k=3)
            else:
                x = random.randint(margin, max(margin, w - probe_w - margin))
                y = random.randint(margin, max(margin, h - probe_h - margin))
            layer = self._decorate(img, text, lang, size, (x, y, probe_w, probe_h), 0.9 * w, 2,
                                   random.choice(["left", "center"]), allow_colour=0.3,
                                   style=random.choice([None, None, "bold", "italic"]))
            layer = _scale_alpha(layer, random.uniform(0.85, 1.0))
            rr = random.random()
            angle = 0.0 if rr < 0.7 else (random.uniform(-20, 20) if rr < 0.9 else random.choice([90.0, -90.0]))
            if abs(angle) > 0.5:
                layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
            layer = _fit_layer(layer, w, h, 0.95)
            x = min(max(0, x), w - layer.width)
            y = min(max(0, y), h - layer.height)
            _composite_at(base, layer, x, y)

        return base.convert("RGB")

    def _caption_outside(self, img: Image.Image, lang: str) -> Image.Image:
        """
        Figure-in-a-document look: a white (or black) strip is added below (or
        above) the image with a caption, optionally with a thin frame / margins.
        """
        w, h = img.size
        text = T = None  # noqa: F841 (readability)
        texts = self.texts
        text = texts.caption(lang) if random.random() < 0.65 else texts.phrase(lang)
        size = self._size(w, h, "caption_outside", lines=2)
        paper = _WHITE if random.random() < 0.85 else random.choice([_BLACK, (245, 245, 240), (30, 30, 30)])
        color = _contrast(paper)
        style = random.choice([None, None, "italic", "bold"])
        layer = self._block(text, lang, size, color, max_w=0.92 * w, max_lines=2,
                            align=random.choice(["center", "center", "left"]), style=style)
        pad = int(size * 0.5)
        border = int(math.sqrt(w * h) * random.uniform(0.0, 0.04)) if random.random() < 0.5 else 0
        strip_h = layer.height + 2 * pad
        above = random.random() < 0.15
        canvas = Image.new("RGB", (w + 2 * border, h + strip_h + 2 * border), paper)
        img_y = border + (strip_h if above else 0)
        canvas.paste(img, (border, img_y))
        if random.random() < 0.4:  # thin frame line around the picture
            d = ImageDraw.Draw(canvas)
            lw = max(1, int(math.sqrt(w * h) * 0.003))
            d.rectangle([border - lw, img_y - lw, border + w + lw - 1, img_y + h + lw - 1], outline=_contrast(paper), width=lw)
        strip_y = border if above else border + h
        x = border + ((w - layer.width) // 2 if layer.width < w else 0)
        if layer.width < w and random.random() < 0.3:
            x = border + pad
        base = canvas.convert("RGBA")
        _composite_at(base, layer, x, strip_y + pad)
        return base.convert("RGB")


# =============================================================================
# Stickers
# =============================================================================

def _draw_star(d: ImageDraw.ImageDraw, cx, cy, r_out, r_in, n, fill, outline, width):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    d.polygon(pts, fill=fill, outline=outline, width=width)


def _centered_text(d: ImageDraw.ImageDraw, font_bank: FontBank, pool: Sequence[str], text, cx, cy, size, max_w, color):
    path, font = font_bank.pick(pool, text, size, style="bold")
    bb = _text_bbox(font, text)
    for _ in range(4):
        if bb[2] - bb[0] <= max_w or size <= 8:
            break
        size *= 0.85 * max_w / max(1, bb[2] - bb[0])
        font = font_bank.load(path, size)
        bb = _text_bbox(font, text)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), text, font=font, fill=color + (255,))


def procedural_sticker(size: int, font_bank: FontBank, text_bank: TextBank) -> Image.Image:
    """Draw a random sticker as RGBA of side ``size`` (rendered at 2x and downsampled)."""
    S = max(16, size * 2)
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    kind = random.choices(
        ["badge", "label", "star", "arrow", "bubble", "heart", "smiley", "qr", "barcode"],
        weights=[3, 3, 2, 2, 2, 1, 2, 1, 1],
    )[0]
    c1 = random.choice(_PALETTE)
    c2 = _contrast(c1)
    lw = max(2, S // 40)
    mg = S // 16
    fill, outline = c1 + (255,), c2 + (255,)
    has_border = random.random() < 0.7
    lang = text_bank.pick_lang()
    pool = text_bank.fonts(lang)

    def short_label() -> str:
        r = random.random()
        if r < 0.5:
            return text_bank.short(lang, 2)
        if r < 0.75:
            return text_bank.number()
        return text_bank.price()

    if kind == "badge":
        if random.random() < 0.6:
            d.ellipse([mg, mg, S - mg, S - mg], fill=fill, outline=outline if has_border else None, width=lw)
        else:
            d.rounded_rectangle([mg, mg, S - mg, S - mg], radius=S // 6, fill=fill, outline=outline if has_border else None, width=lw)
        if random.random() < 0.7:
            _centered_text(d, font_bank, pool, short_label(), S / 2, S / 2, S * 0.28, S * 0.75, c2)
    elif kind == "label":
        hh = S * random.uniform(0.35, 0.6)
        y0 = (S - hh) / 2
        d.rounded_rectangle([mg, y0, S - mg, y0 + hh], radius=int(hh * 0.25), fill=fill, outline=outline if has_border else None, width=lw)
        _centered_text(d, font_bank, pool, short_label(), S / 2, S / 2, hh * 0.5, S * 0.8, c2)
    elif kind == "star":
        n = random.choice([5, 5, 6, 8])
        _draw_star(d, S / 2, S / 2, S / 2 - mg, (S / 2 - mg) * random.uniform(0.45, 0.6), n, fill, outline if has_border else None, lw)
        if random.random() < 0.4:
            _centered_text(d, font_bank, pool, text_bank.number(), S / 2, S / 2 + S * 0.03, S * 0.2, S * 0.45, c2)
    elif kind == "arrow":
        bh = S * random.uniform(0.25, 0.45)
        pts = [(mg, S / 2 - bh / 2), (S * 0.6, S / 2 - bh / 2), (S * 0.6, mg), (S - mg, S / 2),
               (S * 0.6, S - mg), (S * 0.6, S / 2 + bh / 2), (mg, S / 2 + bh / 2)]
        d.polygon(pts, fill=fill, outline=outline if has_border else None, width=lw)
    elif kind == "bubble":
        d.ellipse([mg, mg, S - mg, S * 0.72], fill=fill, outline=outline if has_border else None, width=lw)
        tail = [(S * 0.3, S * 0.65), (S * 0.5, S * 0.65), (S * 0.22, S - mg)]
        d.polygon(tail, fill=fill, outline=outline if has_border else None, width=lw)
        _centered_text(d, font_bank, pool, random.choice(["!", "?", "!!", "?!", text_bank.short(lang, 1)]), S / 2, S * 0.38, S * 0.22, S * 0.7, c2)
    elif kind == "heart":
        r = S * 0.25
        d.ellipse([S * 0.5 - 2 * r + mg / 2, mg, S * 0.5, mg + 2 * r], fill=fill)
        d.ellipse([S * 0.5, mg, S * 0.5 + 2 * r - mg / 2, mg + 2 * r], fill=fill)
        d.polygon([(mg, mg + r * 1.2), (S - mg, mg + r * 1.2), (S / 2, S - mg)], fill=fill)
    elif kind == "smiley":
        yellow = (255, 210, 0)
        d.ellipse([mg, mg, S - mg, S - mg], fill=yellow + (255,), outline=(60, 40, 0, 255), width=lw)
        er = S * 0.06
        for ex in (S * 0.35, S * 0.65):
            d.ellipse([ex - er, S * 0.38 - er, ex + er, S * 0.38 + er], fill=(40, 30, 0, 255))
        if random.random() < 0.7:
            d.arc([S * 0.28, S * 0.35, S * 0.72, S * 0.78], start=20, end=160, fill=(40, 30, 0, 255), width=lw)
        else:
            d.arc([S * 0.28, S * 0.5, S * 0.72, S * 0.9], start=200, end=340, fill=(40, 30, 0, 255), width=lw)
    elif kind == "qr":
        n = random.randint(15, 29)
        cell = S // (n + 2)
        off = (S - cell * n) // 2
        d.rectangle([off - cell, off - cell, off + cell * (n + 1), off + cell * (n + 1)], fill=(255, 255, 255, 255))
        grid = np.random.default_rng(random.getrandbits(32)).random((n, n)) < 0.45
        for i in range(n):
            for j in range(n):
                if grid[i, j]:
                    d.rectangle([off + j * cell, off + i * cell, off + (j + 1) * cell - 1, off + (i + 1) * cell - 1], fill=(0, 0, 0, 255))
        fs = 7 * cell
        for (fx, fy) in [(off, off), (off + (n - 7) * cell, off), (off, off + (n - 7) * cell)]:
            d.rectangle([fx, fy, fx + fs - 1, fy + fs - 1], fill=(0, 0, 0, 255))
            d.rectangle([fx + cell, fy + cell, fx + fs - cell - 1, fy + fs - cell - 1], fill=(255, 255, 255, 255))
            d.rectangle([fx + 2 * cell, fy + 2 * cell, fx + fs - 2 * cell - 1, fy + fs - 2 * cell - 1], fill=(0, 0, 0, 255))
    else:  # barcode
        d.rectangle([mg, S * 0.2, S - mg, S * 0.8], fill=(255, 255, 255, 255))
        x = mg + S // 20
        while x < S - mg - S // 20:
            bw = random.choice([1, 1, 2, 3]) * max(1, S // 120)
            if random.random() < 0.6:
                d.rectangle([x, S * 0.25, x + bw - 1, S * 0.68], fill=(0, 0, 0, 255))
            x += bw + max(1, S // 120)
        _centered_text(d, font_bank, pool, "".join(str(random.randint(0, 9)) for _ in range(12)), S / 2, S * 0.74, S * 0.07, S * 0.8, (0, 0, 0))

    im = im.resize((size, size), Image.LANCZOS)
    bbox = im.split()[-1].getbbox()
    return im.crop(bbox) if bbox else im


class StickerBank:
    """
    Sticker source: a dedicated folder of PNG/WebP files with alpha (never
    training images), mixed with procedurally drawn stickers.
    """

    def __init__(self, sticker_dir: Optional[str], font_bank: FontBank, text_bank: TextBank, asset_prob: float = 0.5):
        self.font_bank = font_bank
        self.text_bank = text_bank
        self.asset_prob = asset_prob
        self.paths: List[str] = []
        if sticker_dir and os.path.isdir(sticker_dir):
            for ext in ("png", "PNG", "webp", "WEBP"):
                self.paths += glob.glob(os.path.join(sticker_dir, "**", f"*.{ext}"), recursive=True)
            self.paths = sorted(set(self.paths))

    def get(self, max_side: int) -> Image.Image:
        """RGBA sticker whose longer side equals ``max_side``."""
        st = None
        if self.paths and random.random() < self.asset_prob:
            p = random.choice(self.paths)
            try:
                st = Image.open(p).convert("RGBA")
                if st.getextrema()[-1][0] == 255 and random.random() < 0.8:
                    st = self._knock_out_background(st)
                bbox = st.split()[-1].getbbox()
                if bbox:
                    st = st.crop(bbox)
            except Exception:
                st = None
        if st is None:
            st = procedural_sticker(max(16, max_side), self.font_bank, self.text_bank)
        s = max_side / max(st.width, st.height, 1)
        nw, nh = max(1, int(st.width * s)), max(1, int(st.height * s))
        return st.resize((nw, nh), Image.LANCZOS)

    @staticmethod
    def _knock_out_background(st: Image.Image, tol: int = 18) -> Image.Image:
        arr = np.asarray(st).astype(np.int16)
        corner = arr[0, 0, :3]
        dist = np.abs(arr[..., :3] - corner).max(axis=-1)
        alpha = np.where(dist <= tol, 0, 255).astype(np.uint8)
        out = arr.astype(np.uint8).copy()
        out[..., 3] = alpha
        return Image.fromarray(out, "RGBA")


def apply_sticker_overlay(image: Image.Image, bank: StickerBank) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    ref = math.sqrt(w * h)
    base = img.convert("RGBA")

    n_stickers = random.choices([1, 2, 3], weights=[7, 2, 1])[0]
    for _ in range(n_stickers):
        target = int(max(ref * random.uniform(0.10, 0.30), 12 * h / NET_SIDE))  # >= 12 px in the network view
        st = bank.get(max(12, target))
        angle = random.uniform(-35, 35) if random.random() < 0.7 else 0.0
        if abs(angle) > 0.5:
            st = st.rotate(angle, expand=True, resample=Image.BICUBIC)
        st = _scale_alpha(st, random.uniform(0.85, 1.0))
        st = _fit_layer(st, w, h, max_frac=0.9)
        sw, sh = st.size
        x = random.randint(0, max(0, w - sw))
        y = random.randint(0, max(0, h - sh))
        if random.random() < 0.5:
            off = max(1, int(target * random.uniform(0.02, 0.05)))
            _composite_at(base, _soft_shadow(st, target * 0.03, 0.5), x + off, y + off)
        _composite_at(base, st, x, y)
    return base.convert("RGB")


# =============================================================================
# Region masks: rect / ellipse / convex polygon / blob
# =============================================================================

REGION_SHAPES: Tuple[str, ...] = ("rect", "ellipse", "convex", "blob")
DEFAULT_REGION_SHAPE_WEIGHTS: Dict[str, float] = {"rect": 1.0, "ellipse": 1.0, "convex": 2.0, "blob": 2.0}


def pick_shape(weights: Optional[Dict[str, float]] = None) -> str:
    w = dict(DEFAULT_REGION_SHAPE_WEIGHTS)
    if weights:
        w.update(weights)
    names = [k for k, v in w.items() if v > 0]
    return random.choices(names, weights=[w[k] for k in names])[0]


def _random_region(w: int, h: int, min_frac: float, max_frac: float,
                   ar_range: Tuple[float, float] = (0.5, 2.0)) -> Tuple[int, int, int, int]:
    area = w * h * random.uniform(min_frac, max_frac)
    ar = random.uniform(*ar_range)
    rw = int(min(0.9 * w, max(8, math.sqrt(area * ar))))
    rh = int(min(0.9 * h, max(8, area / max(rw, 1))))
    rw, rh = min(rw, w), min(rh, h)
    x0 = random.randint(0, max(0, w - rw))
    y0 = random.randint(0, max(0, h - rh))
    return x0, y0, rw, rh


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone chain; returns hull vertices in counter-clockwise order."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _convex_points(box: Tuple[int, int, int, int]) -> List[Tuple[float, float]]:
    """Random convex polygon filling most of the box (hull of 6..14 random points)."""
    x0, y0, rw, rh = box
    n = random.randint(6, 14)
    pts = [(x0 + random.uniform(0, rw), y0 + random.uniform(0, rh)) for _ in range(n)]
    pts += [(x0 + random.uniform(0.2, 0.8) * rw, y0), (x0 + random.uniform(0.2, 0.8) * rw, y0 + rh),
            (x0, y0 + random.uniform(0.2, 0.8) * rh), (x0 + rw, y0 + random.uniform(0.2, 0.8) * rh)]
    hull = _convex_hull(pts)
    return hull if len(hull) >= 3 else [(x0, y0), (x0 + rw, y0), (x0 + rw, y0 + rh), (x0, y0 + rh)]


def _blob_points(box: Tuple[int, int, int, int], n: int = 72) -> List[Tuple[float, float]]:
    """Smooth irregular closed curve (star-shaped around the box centre)."""
    x0, y0, rw, rh = box
    cx, cy = x0 + rw / 2, y0 + rh / 2
    harmonics = [(random.randint(2, 6), random.uniform(0.05, 0.22), random.uniform(0, 2 * math.pi)) for _ in range(random.randint(2, 4))]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        r = 1.0 + sum(amp * math.sin(k * a + ph) for k, amp, ph in harmonics)
        r = max(0.45, min(1.0, r))
        pts.append((cx + r * (rw / 2) * math.cos(a), cy + r * (rh / 2) * math.sin(a)))
    return pts


def _feathered_mask(w: int, h: int, box: Tuple[int, int, int, int], shape: str, feather: float) -> Image.Image:
    """Pixel mask (L) of the edited region inside ``box``; shape: rect | ellipse | convex | blob."""
    x0, y0, rw, rh = box
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    if shape == "ellipse":
        d.ellipse([x0, y0, x0 + rw - 1, y0 + rh - 1], fill=255)
    elif shape == "convex":
        d.polygon(_convex_points(box), fill=255)
    elif shape == "blob":
        d.polygon(_blob_points(box), fill=255)
    else:
        d.rectangle([x0, y0, x0 + rw - 1, y0 + rh - 1], fill=255)
    if feather > 0.5:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


# =============================================================================
# Region edits
# =============================================================================

def _net_min_region(w: int, h: int, min_net_px: float = 10.0) -> float:
    """Minimal region area fraction so that the region is >= min_net_px on both
    axes in the 300x300 network view."""
    return (min_net_px / NET_SIDE) ** 2


def apply_blur_censor(image: Image.Image, shape_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    """Strong blur over a face/plate-sized region (unrecognisable content)."""
    img = image.convert("RGB")
    w, h = img.size
    x0, y0, rw, rh = _random_region(w, h, max(0.02, _net_min_region(w, h, 18)), 0.15, (0.6, 1.7))
    shape = pick_shape(shape_weights)
    region = img.crop((x0, y0, x0 + rw, y0 + rh))
    radius = max(3.0, min(rw, rh) * random.uniform(0.15, 0.35))
    edited = img.copy()
    edited.paste(region.filter(ImageFilter.GaussianBlur(radius)), (x0, y0))
    mask = _feathered_mask(w, h, (x0, y0, rw, rh), shape, min(rw, rh) * random.uniform(0.02, 0.08))
    return Image.composite(edited, img, mask)


def apply_pixelate_censor(image: Image.Image, shape_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    x0, y0, rw, rh = _random_region(w, h, max(0.02, _net_min_region(w, h, 18)), 0.15, (0.6, 1.7))
    region = img.crop((x0, y0, x0 + rw, y0 + rh))
    block = max(3, int(min(rw, rh) * random.uniform(0.1, 0.25)))
    small = region.resize((max(1, rw // block), max(1, rh // block)), Image.BILINEAR)
    edited = img.copy()
    edited.paste(small.resize((rw, rh), Image.NEAREST), (x0, y0))
    mask = _feathered_mask(w, h, (x0, y0, rw, rh), pick_shape(shape_weights), 0.0)
    return Image.composite(edited, img, mask)


def apply_black_box(image: Image.Image, shape_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    """Solid censor bar / box / painted-over patch (eyes bar, plate cover, redaction)."""
    img = image.convert("RGB")
    w, h = img.size
    r = random.random()
    color = _BLACK if r < 0.7 else (_WHITE if r < 0.85 else random.choice([(128, 128, 128)] + _PALETTE))
    if random.random() >= 0.4:  # arbitrary-shaped solid patch
        x0, y0, rw, rh = _random_region(w, h, max(0.02, _net_min_region(w, h, 14)), 0.15, (0.7, 2.0))
        mask = _feathered_mask(w, h, (x0, y0, rw, rh), pick_shape(shape_weights), 0.0)
        return Image.composite(Image.new("RGB", img.size, color), img, mask)
    # thin bar
    x0, y0, rw, rh = _random_region(w, h, max(0.01, _net_min_region(w, h, 8)), 0.06, (3.0, 7.0))
    edited = img.copy()
    ImageDraw.Draw(edited).rectangle([x0, y0, x0 + rw - 1, y0 + rh - 1], fill=color)
    if random.random() < 0.3:  # slightly rotated bar
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle([x0, y0, x0 + rw - 1, y0 + rh - 1], fill=color + (255,))
        layer = layer.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, center=(x0 + rw / 2, y0 + rh / 2))
        base = img.convert("RGBA")
        base.alpha_composite(layer)
        return base.convert("RGB")
    return edited


def apply_copy_move(image: Image.Image, shape_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    """Clone a patch of the same image elsewhere; brightness matched to the destination."""
    img = image.convert("RGB")
    w, h = img.size
    x0, y0, rw, rh = _random_region(w, h, max(0.04, _net_min_region(w, h, 24)), 0.22)
    src = None
    for _ in range(10):
        sx = random.randint(0, max(0, w - rw))
        sy = random.randint(0, max(0, h - rh))
        if abs(sx - x0) >= 0.5 * rw or abs(sy - y0) >= 0.5 * rh:
            src = (sx, sy)
            break
    if src is None:
        src = (0, 0)
    patch = img.crop((src[0], src[1], src[0] + rw, src[1] + rh))
    if random.random() < 0.3:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.8:  # match mean brightness of the destination patch
        dst_mean = np.asarray(img.crop((x0, y0, x0 + rw, y0 + rh)).convert("L"), np.float32).mean()
        src_mean = np.asarray(patch.convert("L"), np.float32).mean()
        gain = float(np.clip(dst_mean / max(src_mean, 1.0), 0.7, 1.4))
        patch = Image.fromarray(np.clip(np.asarray(patch, np.float32) * gain, 0, 255).astype(np.uint8), "RGB")
    edited = img.copy()
    edited.paste(patch, (x0, y0))
    mask = _feathered_mask(w, h, (x0, y0, rw, rh), pick_shape(shape_weights), min(rw, rh) * random.uniform(0.03, 0.08))
    return Image.composite(edited, img, mask)


def apply_inpaint_fill(image: Image.Image, shape_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    """
    Object removal. With OpenCV: real inpainting (Telea / Navier-Stokes) of the
    masked region, computed on a downscaled crop for speed. Without OpenCV: the
    region is filled with a stretched neighbouring band and blended.
    """
    img = image.convert("RGB")
    w, h = img.size
    x0, y0, rw, rh = _random_region(w, h, max(0.03, _net_min_region(w, h, 20)), 0.20)
    shape = pick_shape(shape_weights)
    mask = _feathered_mask(w, h, (x0, y0, rw, rh), shape, 0.0)

    if CV2_AVAILABLE and random.random() < 0.8:
        m = int(min(rw, rh) * 0.5) + 4
        cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
        cx1, cy1 = min(w, x0 + rw + m), min(h, y0 + rh + m)
        crop = img.crop((cx0, cy0, cx1, cy1))
        cmask = mask.crop((cx0, cy0, cx1, cy1))
        scale = min(1.0, 400.0 / max(crop.width, crop.height))  # cap work size
        if scale < 1.0:
            small = crop.resize((max(8, int(crop.width * scale)), max(8, int(crop.height * scale))), Image.BILINEAR)
            smask = cmask.resize(small.size, Image.NEAREST)
        else:
            small, smask = crop, cmask
        arr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
        mk = (np.asarray(smask) > 127).astype(np.uint8) * 255
        radius = max(3, int(min(small.size) * random.uniform(0.02, 0.06)))
        method = cv2.INPAINT_TELEA if random.random() < 0.6 else cv2.INPAINT_NS
        res = cv2.inpaint(arr, mk, radius, method)
        res = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        if scale < 1.0:
            res = res.resize(crop.size, Image.BILINEAR)
        edited = img.copy()
        edited.paste(res, (cx0, cy0))
        soft = mask.filter(ImageFilter.GaussianBlur(max(1.0, min(rw, rh) * 0.03)))
        return Image.composite(edited, img, soft)

    t = max(3, int(min(rw, rh) * random.uniform(0.15, 0.3)))
    sides = []
    if x0 >= t:
        sides.append("left")
    if x0 + rw + t <= w:
        sides.append("right")
    if y0 >= t:
        sides.append("top")
    if y0 + rh + t <= h:
        sides.append("bottom")
    if sides and random.random() < 0.85:
        side = random.choice(sides)
        if side == "left":
            band = img.crop((x0 - t, y0, x0, y0 + rh))
        elif side == "right":
            band = img.crop((x0 + rw, y0, x0 + rw + t, y0 + rh))
        elif side == "top":
            band = img.crop((x0, y0 - t, x0 + rw, y0))
        else:
            band = img.crop((x0, y0 + rh, x0 + rw, y0 + rh + t))
        fillp = band.resize((rw, rh), Image.BILINEAR).filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.5)))
    else:
        region = img.crop((x0, y0, x0 + rw, y0 + rh))
        mean = tuple(int(v) for v in np.asarray(region).reshape(-1, 3).mean(axis=0))
        noise = np.random.default_rng(random.getrandbits(32)).normal(0, random.uniform(3, 10), (rh, rw, 3))
        fillp = Image.fromarray(np.clip(np.asarray(Image.new("RGB", (rw, rh), mean)).astype(np.float32) + noise, 0, 255).astype(np.uint8), "RGB")
        fillp = fillp.filter(ImageFilter.GaussianBlur(random.uniform(1.0, 3.0)))
    edited = img.copy()
    edited.paste(fillp, (x0, y0))
    soft = mask.filter(ImageFilter.GaussianBlur(min(rw, rh) * random.uniform(0.05, 0.12)))
    return Image.composite(edited, img, soft)


def apply_recolor(image: Image.Image, shape_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    """
    Object-aware recolouring: inside a region, only the pixels whose hue is close
    to the region's dominant (saturated) hue get a hue / saturation shift, so the
    result follows an object instead of painting a flat blob.
    """
    img = image.convert("RGB")
    w, h = img.size
    # look for a region that actually contains a coloured object (up to 5 tries)
    for _ in range(5):
        x0, y0, rw, rh = _random_region(w, h, max(0.05, _net_min_region(w, h, 24)), 0.30)
        region = img.crop((x0, y0, x0 + rw, y0 + rh))
        hsv = np.asarray(region.convert("HSV")).astype(np.int16)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        saturated = (sat >= 50) & (val >= 40)
        if saturated.mean() >= 0.15:
            break

    if saturated.mean() >= 0.15:  # hue shift of the dominant colour only
        hist = np.bincount(hue[saturated] // 8, minlength=32)
        dom = int(np.argmax(hist)) * 8 + 4
        dist = np.minimum(np.abs(hue - dom), 256 - np.abs(hue - dom))
        obj = (dist <= random.randint(14, 26)) & (sat >= 35)
        # moderate hue shift (a quarter of the wheel at most) so the result looks
        # like a recoloured object / local colour grading, not a neon blob
        shift = random.choice([-1, 1]) * random.randint(22, 70)
        hsv[..., 0] = (hue + shift) % 256
        hsv[..., 1] = np.clip(sat * random.uniform(0.8, 1.3) + random.randint(-10, 25), 0, 255)
        recol = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    else:  # colourless region (grey, white fur, sky): dodge / burn instead of a colour blob
        obj = np.ones_like(hue, dtype=bool)
        gamma = random.choice([random.uniform(0.45, 0.7), random.uniform(1.5, 2.4)])
        arr = np.asarray(region, np.float32) / 255.0
        recol = Image.fromarray((np.power(arr, gamma) * 255).astype(np.uint8), "RGB")

    obj_mask = Image.fromarray((obj * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.0))
    region_mask = _feathered_mask(rw, rh, (0, 0, rw, rh), pick_shape(shape_weights), min(rw, rh) * random.uniform(0.03, 0.08))
    local = Image.fromarray((np.asarray(obj_mask, np.float32) * np.asarray(region_mask, np.float32) / 255.0).astype(np.uint8), "L")
    full_mask = Image.new("L", (w, h), 0)
    full_mask.paste(local, (x0, y0))
    edited = img.copy()
    edited.paste(recol, (x0, y0))
    return Image.composite(edited, img, full_mask)


def apply_annotation(image: Image.Image, shape_weights: Optional[Dict[str, float]] = None) -> Image.Image:
    """
    Mark-up a person adds to point something out: outline ellipse / rectangle,
    arrow, underline / scribble, highlighter. Colours are the usual red / yellow /
    green / blue markers; line width scales with the image.
    """
    img = image.convert("RGB")
    w, h = img.size
    ref = math.sqrt(w * h)
    base = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    color = random.choice([(230, 30, 30), (230, 30, 30), (255, 220, 0), (40, 180, 60), (30, 110, 230), (255, 120, 0), (0, 0, 0)])
    lw = max(2, int(ref * random.uniform(0.004, 0.012)), int(2 * h / NET_SIDE))  # >= 2 px in the network view
    kind = random.choices(["outline", "arrow", "underline", "highlight", "circle_arrow"], weights=[3, 2, 1.5, 2, 1.5])[0]
    x0, y0, rw, rh = _random_region(w, h, max(0.03, _net_min_region(w, h, 24)), 0.30, (0.5, 2.5))

    def arrow(p0, p1):
        d.line([p0, p1], fill=color + (255,), width=lw)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        hl = lw * random.uniform(3.5, 6)
        for s in (+1, -1):
            a = ang + math.pi - s * math.radians(random.uniform(22, 32))
            d.line([p1, (p1[0] + hl * math.cos(a), p1[1] + hl * math.sin(a))], fill=color + (255,), width=lw)

    if kind in ("outline", "circle_arrow"):
        if random.random() < 0.6:
            d.ellipse([x0, y0, x0 + rw, y0 + rh], outline=color + (255,), width=lw)
        else:
            d.rectangle([x0, y0, x0 + rw, y0 + rh], outline=color + (255,), width=lw)
        if kind == "circle_arrow":
            cx, cy = x0 + rw / 2, y0 + rh / 2
            ang = random.uniform(0, 2 * math.pi)
            far = (cx + math.cos(ang) * (rw + rh), cy + math.sin(ang) * (rw + rh))
            far = (min(max(lw, far[0]), w - lw), min(max(lw, far[1]), h - lw))
            near = (cx + math.cos(ang) * rw * 0.6, cy + math.sin(ang) * rh * 0.6)
            arrow(far, near)
    elif kind == "arrow":
        tip = (x0 + rw / 2, y0 + rh / 2)
        ang = random.uniform(0, 2 * math.pi)
        length = ref * random.uniform(0.15, 0.4)
        tail = (min(max(lw, tip[0] - length * math.cos(ang)), w - lw), min(max(lw, tip[1] - length * math.sin(ang)), h - lw))
        arrow(tail, tip)
    elif kind == "underline":
        y = y0 + rh
        pts = []
        n = random.randint(6, 14)
        for i in range(n):
            x = x0 + rw * i / (n - 1)
            pts.append((x, y + random.uniform(-lw, lw) * (1 if random.random() < 0.6 else 2)))
        d.line(pts, fill=color + (255,), width=lw, joint="curve")
        if random.random() < 0.3:  # double / scribble
            d.line([(px, py + 2.5 * lw) for px, py in pts], fill=color + (255,), width=lw, joint="curve")
    else:  # highlighter
        hc = random.choice([(255, 235, 0), (120, 255, 120), (255, 150, 200), (120, 220, 255)])
        poly = _convex_points((x0, y0, rw, rh)) if random.random() < 0.3 else None
        if poly:
            d.polygon(poly, fill=hc + (random.randint(90, 150),))
        else:
            d.rectangle([x0, y0, x0 + rw, y0 + rh], fill=hc + (random.randint(90, 150),))
    if random.random() < 0.5:  # hand-drawn look: slight blur of the stroke
        layer = layer.filter(ImageFilter.GaussianBlur(lw * 0.15))
    base.alpha_composite(layer)
    return base.convert("RGB")


# =============================================================================
# Observability check
# =============================================================================

def change_stats(before: Image.Image, after: Image.Image, size: int = 192, pixel_thr: int = 12) -> Tuple[float, float]:
    """
    (mean |RGB diff| on 0..255, fraction of pixels whose luminance changed by more
    than ``pixel_thr``) after both images are brought to the same small square,
    i.e. roughly what the network sees after its 300x300 preprocessing.
    A JPEG q95 re-encode gives mean ~0.2-1.2 and a tiny changed fraction.
    """
    a = before.convert("RGB").resize((size, size), Image.BILINEAR)
    b = after.convert("RGB").resize((size, size), Image.BILINEAR)
    fa, fb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    mean = float(np.abs(fa - fb).mean())
    la = fa @ np.array([0.299, 0.587, 0.114], np.float32)
    lb = fb @ np.array([0.299, 0.587, 0.114], np.float32)
    frac = float((np.abs(la - lb) > pixel_thr).mean())
    return mean, frac


def change_magnitude(before: Image.Image, after: Image.Image, size: int = 192) -> float:
    """Mean absolute RGB difference (0..255) at network-like resolution."""
    return change_stats(before, after, size)[0]


# =============================================================================
# Fragment-edit dispatcher (the ``fragment_edit`` token; text is a separate token)
# =============================================================================

FRAGMENT_KINDS: Tuple[str, ...] = (
    "sticker", "annotation", "blur_censor", "pixelate_censor", "black_box",
    "copy_move", "inpaint_fill", "recolor",
)

DEFAULT_FRAGMENT_WEIGHTS: Dict[str, float] = {
    "sticker": 2.0,
    "annotation": 2.0,
    "blur_censor": 1.0,
    "pixelate_censor": 1.0,
    "black_box": 1.0,
    "copy_move": 1.5,
    "inpaint_fill": 1.5,
    "recolor": 1.0,
}


class FragmentEditor:
    """
    Samples a fixed kind of local edit (``FRAGMENT_KINDS``) and applies it.
    Also owns the ``TextOverlayer`` used for the separate ``text_overlay`` token
    (``apply_text``), so fonts and the text bank are loaded once.
    ``last_kind`` / ``last_detail`` describe the most recent edit.
    """

    def __init__(
        self,
        font_bank: Optional[FontBank] = None,
        sticker_dir: Optional[str] = None,
        sticker_asset_prob: float = 0.5,
        text_corpus_dir: Optional[str] = None,
        text_langs: Optional[Sequence[str]] = None,
        text_lang_weights: Optional[Dict[str, float]] = None,
        text_layout_weights: Optional[Dict[str, float]] = None,
        text_min_net_px: float = 9.0,
        kind_weights: Optional[Dict[str, float]] = None,
        region_shapes: Optional[Dict[str, float]] = None,
    ):
        self.font_bank = font_bank or FontBank()
        self.region_shape_weights = dict(DEFAULT_REGION_SHAPE_WEIGHTS)
        if region_shapes:
            for k, v in region_shapes.items():
                if k not in REGION_SHAPES:
                    raise ValueError(f"unknown region shape '{k}', allowed: {REGION_SHAPES}")
                self.region_shape_weights[k] = float(v)
        if not any(v > 0 for v in self.region_shape_weights.values()):
            raise ValueError("all region shapes have zero weight")

        self.text_bank = TextBank(self.font_bank, text_corpus_dir, text_langs, text_lang_weights)
        self.text_overlayer = TextOverlayer(self.font_bank, self.text_bank, text_layout_weights, min_net_px=text_min_net_px)
        self.sticker_bank = StickerBank(sticker_dir, self.font_bank, self.text_bank, sticker_asset_prob)

        self.kind_weights = dict(DEFAULT_FRAGMENT_WEIGHTS)
        if kind_weights:
            for k, v in kind_weights.items():
                if k not in FRAGMENT_KINDS:
                    raise ValueError(f"unknown fragment kind '{k}', allowed: {FRAGMENT_KINDS}")
                self.kind_weights[k] = float(v)
        self.kind_weights = {k: v for k, v in self.kind_weights.items() if v > 0}
        if not self.kind_weights:
            raise ValueError("all fragment kinds have zero weight")

        self.last_kind: Optional[str] = None
        self.last_detail: Optional[str] = None

    def sample_kind(self, exclude: Sequence[str] = ()) -> str:
        kinds = [k for k in self.kind_weights if k not in exclude] or list(self.kind_weights)
        return random.choices(kinds, weights=[self.kind_weights[k] for k in kinds])[0]

    def apply_text(self, image: Image.Image, layout: Optional[str] = None) -> Tuple[Image.Image, str]:
        """``text_overlay`` token: returns (image, "layout/lang")."""
        out = self.text_overlayer.apply(image, layout)
        detail = f"{self.text_overlayer.last_layout}/{self.text_overlayer.last_lang}"
        self.last_kind, self.last_detail = "text", detail
        return out, detail

    def apply(self, image: Image.Image, kind: Optional[str] = None) -> Tuple[Image.Image, str]:
        """``fragment_edit`` token: one local edit of the given / sampled kind."""
        kind = kind or self.sample_kind()
        sw = self.region_shape_weights
        if kind == "sticker":
            out = apply_sticker_overlay(image, self.sticker_bank)
        elif kind == "annotation":
            out = apply_annotation(image, sw)
        elif kind == "blur_censor":
            out = apply_blur_censor(image, sw)
        elif kind == "pixelate_censor":
            out = apply_pixelate_censor(image, sw)
        elif kind == "black_box":
            out = apply_black_box(image, sw)
        elif kind == "copy_move":
            out = apply_copy_move(image, sw)
        elif kind == "inpaint_fill":
            out = apply_inpaint_fill(image, sw)
        elif kind == "recolor":
            out = apply_recolor(image, sw)
        else:
            raise ValueError(f"unknown fragment kind '{kind}'")
        self.last_kind, self.last_detail = kind, None
        return out, kind
