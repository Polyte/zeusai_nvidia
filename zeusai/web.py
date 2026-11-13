"""Keyless web access for research mode: DuckDuckGo HTML search and page fetch."""
import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
MAX_PAGE_CHARS = 15000


def _clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def _real_url(href):
    # DuckDuckGo sometimes wraps results as //duckduckgo.com/l/?uddg=<url>
    if "uddg=" in href:
        return urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlsplit(href).query).get("uddg", [href])[0])
    return "https:" + href if href.startswith("//") else href


def search(query, max_results=6):
    req = urllib.request.Request("https://html.duckduckgo.com/html/",
                                 data=urllib.parse.urlencode({"q": query}).encode(),
                                 headers={"User-Agent": UA})
    page = urllib.request.urlopen(req, timeout=15).read().decode(errors="replace")
    links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    hits = []
    for i, (href, title) in enumerate(links):
        url = _real_url(html.unescape(href))
        if "duckduckgo.com/y.js" in url:  # ads
            continue
        hits.append({"title": _clean(title), "url": url, "snippet": _clean(snippets[i]) if i < len(snippets) else ""})
        if len(hits) >= max_results:
            break
    return hits


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "section", "article"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def fetch(url):
    if not url.startswith(("http://", "https://")):
        return "Error: only http(s) URLs are supported"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,text/plain,*/*"})
    with urllib.request.urlopen(req, timeout=20) as r:
        ctype = r.headers.get_content_type()
        raw = r.read(3_000_000).decode(r.headers.get_content_charset() or "utf-8", errors="replace")
    if ctype == "text/html":
        p = _Text()
        p.feed(raw)
        raw = "".join(p.out)
    text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", raw)).strip()
    return text[:MAX_PAGE_CHARS] + ("\n... (truncated)" if len(text) > MAX_PAGE_CHARS else "")
