from __future__ import annotations
import os, re, time, hashlib, argparse, sys, html
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception as e:
    raise RuntimeError(
        "crawl.py の実行には 'requests' と 'beautifulsoup4' が必要です。\n"
        "例: pip install requests beautifulsoup4"
    ) from e


# ==== 設定（必要に応じて調整） ==============================================
DEFAULT_OUTDIR = "docs/web_scraped"
USER_AGENT = "kokosan-bot/0.1 (+https://example.local)"
TIMEOUT = 12
DEFAULT_MAX_PAGES = 30
DEFAULT_DELAY_SEC = 0.5

DEFAULT_ALLOWED = {
    "仕事・職場": ["check-roudou.mhlw.go.jp"],
    "お金・ライフプラン": ["fsa.go.jp", "toushin.or.jp", "ideco-koushiki.jp", "j-flec.go.jp"],
}

DEFAULT_SEEDS = {
    "仕事・職場": ["https://www.check-roudou.mhlw.go.jp/"],
    "お金・ライフプラン": ["https://www.fsa.go.jp/policy/nisa2/index.html", "https://www.ideco-koushiki.jp/", "https://www.j-flec.go.jp/", "https://www.toushin.or.jp/investmenttrust/about/what/"],
    "その他・雑談": []
}

# ==== ユーティリティ =========================================================
def norm_url(url: str) -> str:
    """フラグメント除去＆末尾スラッシュ正規化"""
    url, _ = urldefrag(url)
    if url.endswith("/"):
        return url[:-1]
    return url

def is_allowed(url: str, allow_domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(dom in host for dom in allow_domains)

def same_site(a: str, b: str) -> bool:
    pa, pb = urlparse(a), urlparse(b)
    return pa.scheme == pb.scheme and pa.netloc == pb.netloc

def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_main_text(soup: BeautifulSoup) -> tuple[str, str]:
    # タイトル
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # 本文は保守的に p/li/td/h1-h3 を抽出
    parts = []
    for sel in ["h1", "h2", "h3", "p", "li", "td"]:
        for t in soup.select(sel):
            txt = t.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
    text = clean_text("\n".join(parts))
    return title, text

def save_markdown(outdir: str, category: str, url: str, title: str, text: str) -> str:
    os.makedirs(os.path.join(outdir, category), exist_ok=True)
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    path = os.path.join(outdir, category, f"{h}.md")
    fm = [
        "---",
        f"source: {url}",
        f"title: {title[:200]}",
        f"category: {category}",
        f"fetched_at: {datetime.utcnow().isoformat()}Z",
        "---",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(fm))
        f.write(text + "\n")
    return path


# ==== クロール本体 ===========================================================
def crawl_category(
    category: str,
    seeds: list[str],
    allowed_domains: list[str],
    outdir: str = DEFAULT_OUTDIR,
    max_pages: int = DEFAULT_MAX_PAGES,
    delay_sec: float = DEFAULT_DELAY_SEC,
) -> dict:
    """カテゴリ単位でクロールし、.md を保存していく"""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    seen: set[str] = set()
    saved = 0
    q = deque(norm_url(u) for u in seeds if is_allowed(u, allowed_domains))

    while q and len(seen) < max_pages:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        try:
            resp = session.get(url, timeout=TIMEOUT)
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            continue

        title, text = extract_main_text(soup)
        if len(text) >= 300:  # 短すぎるページは除外
            save_markdown(outdir, category, url, title or url, text)
            saved += 1

        # 内部リンクを少し追加（同一サイト・同一ドメインのみ）
        for a in soup.select("a[href]"):
            nxt = urljoin(url, a["href"])
            nxt = norm_url(nxt)
            if not nxt.startswith(("http://", "https://")):
                continue
            if not is_allowed(nxt, allowed_domains):
                continue
            if not same_site(url, nxt):
                # ドメインが許可されていても、まずは同一サイト内に限定（暴走防止）
                continue
            if nxt not in seen:
                q.append(nxt)

        time.sleep(delay_sec)

    return {"category": category, "visited": len(seen), "saved": saved}


# ==== CLI ====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="ホワイトリスト付きミニクロール → .md保存")
    p.add_argument("--category", choices=list(DEFAULT_ALLOWED.keys()), required=True,
                   help="対象カテゴリ（仕事・職場 / お金・ライフプラン / その他・雑談）")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--max_pages", type=int, default=DEFAULT_MAX_PAGES)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY_SEC)
    p.add_argument("--seed", action="append", help="追加のシードURL（複数可）")
    return p.parse_args()

def main():
    args = parse_args()
    category = args.category
    allowed = DEFAULT_ALLOWED.get(category, [])
    seeds = list(DEFAULT_SEEDS.get(category, []))
    if args.seed:
        seeds.extend(args.seed)
    if not allowed or not seeds:
        print(f"[CRAWL] 設定不足: category={category} allowed={allowed} seeds={seeds}")
        sys.exit(1)

    print(f"[CRAWL] category={category} seeds={len(seeds)} allowed={allowed}")
    stats = crawl_category(
        category=category,
        seeds=seeds,
        allowed_domains=allowed,
        outdir=args.outdir,
        max_pages=args.max_pages,
        delay_sec=args.delay,
    )
    print(f"[CRAWL] done: {stats}")

if __name__ == "__main__":
    main()