import re
from collections import Counter

def clean_text(s):
    s = str(s).upper()
    s = s.replace(" ", "").replace("_", "-")
    s = re.sub(r"[^A-Z0-9\-.]", "", s)
    return s.replace("..", ".").replace("--", "-")

def clean_num(s):
    s = clean_text(s)
    table = str.maketrans({"O":"0", "Q":"0", "D":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "B":"8", "G":"6"})
    return s.translate(table)

def fix_province(s):
    table = str.maketrans({"O":"0", "Q":"0", "D":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "B":"8", "G":"6"})
    return s[:2].translate(table) + s[2:]

def score_text(s):
    s = clean_text(s)
    return len(re.findall(r"\d", s)) * 4 + len(re.findall(r"[A-Z]", s)) * 3 + len(s)

def expand_cands(cands):
    base = [clean_text(x) for x in cands if clean_text(x)]
    nums = [clean_num(x) for x in cands if clean_num(x)]
    prefs = [x.strip("-.") for x in base if re.fullmatch(r"\d{2}[-.]?", x)]
    sufs = [x.strip("-.") for x in base if re.fullmatch(r"[A-Z0-9]{1,3}", x) and not re.fullmatch(r"\d{3}", x)]
    extra = [f"{p}-{s}" for p in prefs for s in sufs if p != s]
    return base + nums + extra

def norm_top(raw):
    s0 = clean_text(raw)
    if re.fullmatch(r"\d{4,}", re.sub(r"\D", "", s0)) and not re.search(r"[A-Z]", s0):
        return None
    s = fix_province(s0.replace(".", "-"))
    m = re.fullmatch(r"(\d{2})-?([A-Z0-9]{1,3})", s)
    if not m:
        return None
    prov, body = m.groups()
    if body.isdigit() and len(body) > 2:
        return None
    return f"{prov}-{body}"

def norm_bottom(raw):
    s = clean_num(raw)
    if re.search(r"[A-Z]", s):
        return None
    nums = re.sub(r"\D", "", s)
    if len(nums) < 4:
        return None
    nums = nums[:5]
    if len(nums) == 5:
        return nums[:3] + "." + nums[3:]
    return nums

def pick_best_top(cands, fallback):
    pool = expand_cands(cands)
    normed = [norm_top(x) for x in pool]
    normed = [x for x in normed if x]

    if not normed:
        pool = expand_cands(fallback)
        normed = [norm_top(x) for x in pool]
        normed = [x for x in normed if x]

    if not normed:
        return ""

    cnt = Counter(normed)

    def rank(x):
        body = x.split("-")[-1]
        return cnt[x] * 1500 + ("-" in x) * 120 + bool(re.search(r"[A-Z]", body)) * 100 + len(body) * 10

    return sorted(set(normed), key=rank, reverse=True)[0]

def pick_best_bottom(cands, fallback):
    pool = expand_cands(cands)
    normed = [norm_bottom(x) for x in pool]
    normed = [x for x in normed if x]

    if not normed:
        pool = expand_cands(fallback)
        normed = [norm_bottom(x) for x in pool]
        normed = [x for x in normed if x]

    if not normed:
        return ""

    cnt = Counter(normed)

    def rank(x):
        nums = re.sub(r"\D", "", x)
        return cnt[x] * 1500 + len(nums) * 100 + ("." in x) * 20

    return sorted(set(normed), key=rank, reverse=True)[0]

def pick_plate(top_cands, bottom_cands, all_cands):
    top = pick_best_top(top_cands, all_cands)
    bot = pick_best_bottom(bottom_cands, all_cands)

    show = expand_cands(all_cands + top_cands + bottom_cands)
    cnt = Counter([clean_text(x) for x in show if clean_text(x)])
    show = sorted(set(cnt), key=lambda x: cnt[x] * 1000 + score_text(x), reverse=True)

    if top and bot:
        return f"{top}\n{bot}", show[:20]
    if bot:
        return bot, show[:20]
    if top:
        return top, show[:20]
    return (show[0] if show else ""), show[:20]