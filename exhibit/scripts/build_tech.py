#!/usr/bin/env python3
"""Build the server-free technical explanation page (assets/tech.html).

Every setting shown on the page is read from the same files the exhibit runs
on (configs/demo.json, configs/fan-policies.json, assets/manifest.json and the
reviewed catalog), so the explanation cannot drift from the implementation.
Fonts are embedded and nothing is requested from the network, so the page also
opens from file://.
"""

import base64
import html
from pathlib import Path

from exhibit.catalog import load_catalog
from exhibit.config import ASSETS, CONFIG, FAN_POLICIES, ROOT, read_json
from exhibit.domain import GAINS, LEGACY_POLICY_ID, STRENGTHS
from exhibit.fan_adapter import profiling_argument, resolve_policy, thaw_policy

STATIC = ROOT / "src/exhibit/static"
OUTPUT = ASSETS / "tech.html"

# Server paths, with the file:// equivalent relative to exhibit/assets/.
LINKS = {
    "home": ("/", None),
    "fallback": ("/fallback", "fallback.html"),
    "report": ("/report/", "../../docs/reports/fan-personalization/index.html"),
}
FAN_REPO = "https://github.com/Burf/FAN"
MODEL_PAGE = "https://huggingface.co/{}"


def e(value):
    return html.escape(str(value))


def number(value):
    """1.0 -> "1", 0.5 -> "0.5": the way the values are written in the configs."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def font_face(family, filename):
    encoded = base64.b64encode((STATIC / filename).read_bytes()).decode()
    return (
        f'@font-face{{font-family:"{family}";font-weight:100 900;font-display:swap;'
        f'src:url(data:font/woff2;base64,{encoded}) format("woff2")}}'
    )


def link(name, label, cls=""):
    href, local = LINKS[name]
    attrs = f' class="{cls}"' if cls else ""
    if local:
        attrs += f' data-file="{e(local)}"'
    else:
        attrs += " data-server-only"
    return f'<a href="{e(href)}"{attrs}>{label}</a>'


def sampler_label(generation):
    """The common name of the configured scheduler, derived from its kwargs."""
    kwargs = generation.get("scheduler_kwargs", {})
    if (
        generation["scheduler"] == "DPMSolverMultistepScheduler"
        and kwargs.get("algorithm_type") == "sde-dpmsolver++"
        and kwargs.get("solver_order", 2) == 2
    ):
        return "DPM++ 2M SDE" + (" Karras" if kwargs.get("use_karras_sigmas") else "")
    return generation["scheduler"]


def layer_range(indices):
    indices = sorted(indices)
    if indices and indices == list(range(indices[0], indices[-1] + 1)):
        return f"{indices[0]}〜{indices[-1]}"
    return ", ".join(str(index) for index in indices)


STYLE = """
:root{color-scheme:dark;--bg:#100d0c;--panel:#191514;--raise:#221c1a;--line:#30282a;
--ink:#f8f1eb;--sub:#ab9f97;--mute:#75695f;--a1:#ff9a6b;--a2:#ff6f93;--on-accent:#1a0e0a;
--accent:linear-gradient(135deg,var(--a1),var(--a2));
--tint:color-mix(in srgb,var(--a1) 12%,transparent);
--sans:"Geist","Noto Sans CJK JP","Noto Sans JP","Hiragino Kaku Gothic ProN","Yu Gothic",system-ui,sans-serif;
--mono:"Geist Mono","SFMono-Regular",Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;
line-height:1.85;-webkit-font-smoothing:antialiased;overflow-wrap:break-word}
h1,h2,h3,p,ol,ul,dl,dd,figure{margin:0}
ol,ul{padding:0;list-style:none}
a{color:inherit;text-underline-offset:3px}
a:focus-visible{outline:2px solid var(--a1);outline-offset:3px}
code{font-family:var(--mono);font-size:.86em;color:var(--ink);background:var(--raise);
padding:1px 6px;border-radius:6px;overflow-wrap:anywhere}
header{display:flex;align-items:center;flex-wrap:wrap;gap:10px 18px;
padding:16px max(16px,4vw);border-bottom:1px solid var(--line)}
.brand{display:inline-flex;align-items:center;gap:10px;font-weight:700;letter-spacing:.04em;
text-decoration:none}
.mark{width:28px;height:28px;display:grid;place-items:center;border-radius:8px;background:var(--accent)}
.mark svg{width:18px;height:18px;stroke:var(--on-accent)}
.powered{font-size:13px;color:var(--mute);padding-left:14px;border-left:1px solid var(--line)}
nav{display:flex;flex-wrap:wrap;gap:6px 16px;margin-left:auto;font-size:13px}
nav a{color:var(--sub);text-decoration:none}nav a:hover{color:var(--ink)}
main{max-width:880px;margin:auto;padding:48px 16px 72px}
.pill{display:inline-block;padding:4px 14px;border-radius:999px;background:var(--tint);
border:1px solid color-mix(in srgb,var(--a1) 35%,transparent);color:var(--a1);font-size:13px;font-weight:600}
h1{margin-top:16px;font-size:clamp(30px,5.2vw,48px);font-weight:900;line-height:1.3;letter-spacing:-.02em}
.grad{background:var(--accent);-webkit-background-clip:text;background-clip:text;color:transparent}
.lead{margin-top:16px;color:var(--sub);max-width:44em}
p{color:var(--sub)}p+p{margin-top:12px}
strong{color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}
.chip{padding:6px 12px;border:1px solid var(--line);border-radius:10px;background:var(--panel);font-size:13px;color:var(--sub)}
.chip b{font-family:var(--mono);font-weight:500;color:var(--ink);margin-left:6px}
section{margin-top:72px}
h2{display:flex;align-items:baseline;gap:12px;font-size:clamp(22px,3.2vw,28px);font-weight:900;
line-height:1.4;letter-spacing:-.01em;margin-bottom:16px}
h2 .n{flex:none;font-family:var(--mono);font-size:14px;font-weight:600;color:var(--on-accent);
background:var(--accent);border-radius:8px;padding:2px 9px}
h3{font-size:17px;margin:28px 0 8px}
.steps{position:relative;margin-top:24px}
.steps li{position:relative;padding:0 0 22px 48px}
.steps li:before{content:attr(data-n);position:absolute;left:0;top:0;width:32px;height:32px;
display:grid;place-items:center;border-radius:50%;background:var(--accent);color:var(--on-accent);
font-family:var(--mono);font-weight:700;font-size:14px}
.steps li:not(:last-child):after{content:"";position:absolute;left:15px;top:36px;bottom:4px;width:2px;background:var(--line)}
.steps b{display:block;color:var(--ink)}
.steps span{color:var(--sub);font-size:14px}
.split{margin-top:12px;border:1px solid var(--line);border-radius:20px;background:var(--panel);padding:18px}
.shared{text-align:center;border:1px dashed var(--mute);border-radius:14px;padding:12px;font-size:14px;color:var(--sub)}
.shared b{color:var(--ink)}
.fork{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.lane{position:relative;border-radius:14px;padding:14px 16px;background:var(--raise);border:1px solid var(--line)}
.lane:before{content:"↓";position:absolute;top:-14px;left:50%;transform:translateX(-50%);
color:var(--mute);font-size:13px;line-height:1}
.lane.personal{border-color:color-mix(in srgb,var(--a2) 55%,transparent);background:linear-gradient(160deg,var(--tint),var(--raise) 70%)}
.lane h4{margin:0 0 6px;font-size:15px}
.lane.personal h4{color:var(--a1)}
.lane ul li{font-size:13px;color:var(--sub);padding-left:14px;position:relative}
.lane ul li:before{content:"·";position:absolute;left:2px}
.note{margin-top:12px;font-size:13px;color:var(--mute)}
.callout{margin-top:20px;padding:16px 18px;border-radius:14px;background:var(--tint);
border-left:3px solid var(--a1);color:var(--ink);font-size:14px}
.callout p{color:var(--ink)}
.formula{margin:18px 0;padding:16px;border-radius:14px;background:var(--panel);border:1px solid var(--line);
font-family:var(--mono);font-size:14px;text-align:center;color:var(--ink);line-height:2}
.formula small{display:block;font-family:var(--sans);font-size:12px;color:var(--mute)}
dl.kv{display:grid;grid-template-columns:minmax(150px,220px) 1fr;border-top:1px solid var(--line);margin-top:16px}
dl.kv dt,dl.kv dd{padding:12px 0;border-bottom:1px solid var(--line)}
dl.kv dt{color:var(--ink);font-weight:600;padding-right:16px}
dl.kv dd{color:var(--sub);font-size:14px}
dl.kv dd small{display:block;color:var(--mute);font-size:12.5px;margin-top:2px}
.card-example{margin-top:16px;border:1px solid var(--line);border-radius:14px;overflow:hidden}
.card-example div{display:grid;grid-template-columns:90px 1fr;gap:12px;padding:10px 14px;border-top:1px solid var(--line);font-size:13.5px}
.card-example div:first-child{border-top:0;background:var(--panel);color:var(--ink);display:block}
.card-example span{color:var(--mute)}
.card-example code{background:none;padding:0;color:var(--ink)}
ul.dont li{position:relative;padding:12px 0 12px 30px;border-bottom:1px solid var(--line);color:var(--sub)}
ul.dont li:before{content:"×";position:absolute;left:4px;top:11px;color:var(--a2);font-weight:700}
ul.dont b{color:var(--ink)}
.more{margin-top:16px;font-size:14px}
.more a{color:var(--a1)}
footer{border-top:1px solid var(--line);padding:20px max(16px,4vw);font-size:12px;color:var(--mute);
display:flex;flex-wrap:wrap;gap:8px 20px;justify-content:space-between}
footer a{color:var(--sub)}footer .links{display:flex;flex-wrap:wrap;gap:4px 16px}
@media(max-width:640px){main{padding:32px 16px 56px}.powered{display:none}nav{margin-left:0;width:100%}
.fork{grid-template-columns:1fr}.lane:before{display:none}
dl.kv{grid-template-columns:1fr}dl.kv dt{border-bottom:0;padding-bottom:0}
.card-example div{grid-template-columns:1fr;gap:0}}
"""

# On file:// the server paths do not exist: point to the sibling files instead.
FILE_LINKS = (
    '<script>if(location.protocol==="file:"){'
    'for(const a of document.querySelectorAll("a[data-file]"))a.href=a.dataset.file;'
    'for(const a of document.querySelectorAll("a[data-server-only]"))a.hidden=true}'
    "</script>"
)

MARK = (
    '<span class="mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" '
    'stroke-linecap="round"><path d="M6 19V6M6 19l9.2-9.2M6 19h13" stroke-width="2.4"/>'
    '<path d="M6 6a13 13 0 0 1 13 13" stroke-width="1.6"/></svg></span>'
)
ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
    "%3Crect width='24' height='24' rx='6' fill='%23ff8a7a'/%3E%3Cg fill='none' "
    "stroke='%231a0e0a' stroke-linecap='round'%3E%3Cpath d='M6 19V6M6 19l9.2-9.2M6 19h13' "
    "stroke-width='2.4'/%3E%3Cpath d='M6 6a13 13 0 0 1 13 13' stroke-width='1.6'/%3E"
    "%3C/g%3E%3C/svg%3E"
)


def facts():
    """Every value on the page, each read from where the exhibit itself reads it."""
    generation = CONFIG["generation"]
    manifest = read_json(ASSETS / "manifest.json", {}) or {}
    catalog_config = read_json(ROOT / "configs/catalog-v2.json", {}) or {}
    catalog = load_catalog(reviewed_only=True)
    default_id = FAN_POLICIES["default_policy_id"]
    if default_id != LEGACY_POLICY_ID:
        # The reasons in section 4 were measured for legacy_exhibit only.
        raise SystemExit(
            f"Section 4 explains {LEGACY_POLICY_ID}; update build_tech.py for {default_id}"
        )
    return {
        "generation": generation,
        "model_name": generation["model"].split("/")[-1].replace("-", " "),
        "sampler": sampler_label(generation),
        "seeds": CONFIG["seeds"],
        "timeout": CONFIG["timeout_seconds"],
        "idle": CONFIG["idle_seconds"],
        "selection": CONFIG["selection"],
        "aspects": CONFIG["aspect_labels"],
        "topics": CONFIG["topics"],
        "fan": CONFIG["fan"],
        "default_id": default_id,
        "policy": thaw_policy(resolve_policy(default_id, FAN_POLICIES)),
        "other_policies": sorted(set(FAN_POLICIES["policies"]) - {default_id}),
        "manifest_matches": manifest.get("generation") == generation,
        "generic_count": len(manifest.get("images", {})),
        "catalog_id": catalog["catalog_id"],
        "card_total": len(catalog["all_cards"]),
        "subjects": len(catalog_config.get("subjects", [])),
        "profiles": len(catalog_config.get("profiles", [])),
        "example": catalog["cards"][0] if catalog["cards"] else None,
    }


def render(f=None):
    f = f or facts()
    g, policy, sel = f["generation"], f["policy"], f["selection"]
    aspects = f["aspects"]
    aspect_names = "・".join(aspects.values())
    alpha = number(policy["alpha"])
    size = f"{g['width']}×{g['height']}"
    seeds = len(f["seeds"])
    commit = f["fan"]["commit"]
    decoders = f["fan"]["decoders"]
    skip_pa = layer_range(policy["skip_pa"])
    sample_size = profiling_argument(policy)
    strengths = " / ".join(number(value) for value in STRENGTHS)
    gains = " / ".join(number(value) for value in GAINS)
    vae = g.get("vae") or {}
    model = e(f["model_name"])
    parts = [
        (
            '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="theme-color" content="#100d0c">'
            "<title>しくみ · パーソナライズ画像生成</title>"
            f'<link rel="icon" href="{ICON}">'
            "<style>"
            + font_face("Geist", "Geist-Variable.woff2")
            + font_face("Geist Mono", "GeistMono-Variable.woff2")
            + " ".join(STYLE.split("\n"))
            + "</style></head><body>"
            f'<header><span class="brand">{MARK}パーソナライズ画像生成</span>'
            '<span class="powered">powered by FAN</span><nav>'
            + link("home", "体験に戻る")
            + link("fallback", "事前生成サンプル")
            + link("report", "検証記録")
            + "</nav></header><main>"
            # ------------------------------------------------------------ hero
            '<span class="pill">技術解説</span>'
            '<h1>好みを選ぶだけで、<span class="grad">画像が変わるしくみ</span></h1>'
            '<p class="lead">この展示の実装が、どう動いているかを1ページにまとめました。'
            "何を測って何が確認できたか（検証の記録と数値）は "
            + link("report", "検証記録")
            + " にあります。ここでは数値は要点だけにしています。</p>"
            '<div class="chips">'
            f'<span class="chip">モデル<b>{model}</b></span>'
            f'<span class="chip">解像度<b>{size}</b></span>'
            f'<span class="chip">steps<b>{g["steps"]}</b></span>'
            f'<span class="chip">sampler<b>{e(f["sampler"])}</b></span>'
            f'<span class="chip">CFG<b>{g["guidance_scale"]}</b></span>'
            f'<span class="chip">alpha<b>{alpha}</b></span>'
            "</div>"
            # ------------------------------------------------------------ 1
            '<section id="flow"><h2><span class="n">1</span>全体の流れ</h2>'
            "<p>来場者が決めるのは「どの画像のどこが好きか」だけです。"
            "それが参照の文になり、テキストエンコーダーの段階でお題の文に混ざります。</p>"
            '<ol class="steps">'
            f'<li data-n="1"><b>好きな画像を選ぶ</b><span>1画面{sel["round_size"]}枚・'
            f"最大{sel['max_rounds']}回で、合計{sel['min']}〜{sel['max']}枚。</span></li>"
            f'<li data-n="2"><b>好きな側面を指定する</b><span>選んだ画像ごとに'
            f"{e(aspect_names)}から1つ以上。強さ <code>strength</code> は {strengths}。</span></li>"
            '<li data-n="3"><b>確認済みの説明文を参照にする</b><span>'
            "指定した側面に付いた短い英語の句を集めて、重み付きの参照リストにします。</span></li>"
            '<li data-n="4"><b>FAN でお題の文をエンコードする</b><span>'
            "お題の文と参照リストを一緒にテキストエンコーダーへ入れ、好みを混ぜた埋め込みを作ります。</span></li>"
            '<li data-n="5"><b>SDXL で生成する</b><span>'
            f"{model} が {seeds}つの seed で1枚ずつ描きます。</span></li>"
            "</ol>"
            '<figure class="split" aria-label="通常生成とパーソナライズ生成の比較の構成">'
            '<div class="shared"><b>共通</b>：お題の文 ・ 負のプロンプト ・ '
            f"seed {seeds}つ ・ 生成モデルと生成設定</div>"
            '<div class="fork">'
            '<div class="lane"><h4>通常生成</h4><ul>'
            "<li>参照なしでエンコード</li>"
            "<li>お題ごとに事前生成した画像を表示</li></ul></div>"
            '<div class="lane personal"><h4>パーソナライズ生成</h4><ul>'
            "<li>参照ありで FAN エンコード</li>"
            "<li>その場で生成し、できた順に表示</li></ul></div>"
            "</div>"
            '<figcaption class="note">左右で違うのは、参照の内容・重み <code>weight</code>・'
            "反映の強さ <code>alpha</code> だけです。同じ seed の順に並べるので、"
            "同じ位置の2枚は同じ seed から描かれています。</figcaption></figure>"
            "</section>"
            # ------------------------------------------------------------ 2
            '<section id="fan"><h2><span class="n">2</span>FAN の仕組み</h2>'
            f'<p><a href="{FAN_REPO}">FAN</a>（Foundation Encoders Are All You Need for '
            "Preference-Aware Personalization, CVPR 2026）の公式実装を、固定 commit "
            f"<code>{e(commit[:12])}</code> のまま使っています。"
            "SDXL の2つのテキストエンコーダー（CLIP-L と OpenCLIP bigG）の両方で、"
            "self-attention を <strong>personalized attention</strong> に差し替えます。</p>"
            '<div class="formula">お題のトークンの注意 = (1 − α) × お題の文 ＋ α × 参照の文'
            "<small>参照どうしの配分は参照ごとの weight で決まり、参照全体に回る割合は α で決まります</small></div>"
            "<p>お題の文のトークンは、自分の文に加えて参照の文のトークンにも注意を向けます。"
            "その結果の hidden states を <code>prompt_embeds</code> として生成モデルへ渡します。"
            f"展示の α は <strong>{alpha}</strong> です。"
            "weight を大きくしても、ほかの参照との比率が変わるだけで、参照全体の強さは α のままです。</p>"
            "<h3>ClassTokenDecoder（追加の重み）</h3>"
            "<p>参照を混ぜた系列から pooled 埋め込みを取るトークン位置を選ぶ小さな分類器で、"
            "公式リポジトリ同梱の <code>weight/L.pth</code> と <code>weight/bigG.pth</code> を読み込みます"
            f"（sha256 は設定に固定：L <code>{e(decoders['L.pth'][:12])}…</code> / "
            f"bigG <code>{e(decoders['bigG.pth'][:12])}…</code>）。"
            "展示の既定設定では pooled を参照なしの値に置き換えるため（4章）、"
            "画像に効いているのは hidden states 側の混合です。</p>"
            '<div class="callout"><p><strong>来場者ごとの追加学習はしません。</strong>'
            "生成モデルもテキストエンコーダーも更新しません。"
            "ただし「追加の重みが一切ない」わけではなく、FAN 公式の ClassTokenDecoder の重みは使っています。</p></div>"
            "</section>"
        )
    ]
    # ---------------------------------------------------------------- 3
    example = f["example"]
    example_rows = ""
    if example:
        example_rows = (
            '<div class="card-example"><div>例：確認済みカード「'
            f"{e(example['label'])}」に付いている句</div>"
            + "".join(
                f"<div><span>{e(label)}</span><code>{e(example['aspects'][key])}</code></div>"
                for key, label in aspects.items()
                if key in example["aspects"]
            )
            + "</div>"
            f'<p class="note">被写体「{e(example["subject_label"])}」を表す語は、句に含まれていません。</p>'
        )
    parts.append(
        '<section id="refs"><h2><span class="n">3</span>参照の作り方</h2>'
        "<p>参照にするのは<strong>画像そのものではなく、画像に付けた説明文</strong>です。"
        f"カードは <code>{e(f['catalog_id'])}</code> の {f['card_total']}枚"
        f"（{f['subjects']}被写体 × {f['profiles']}表現）で、"
        f"{e(aspect_names)}の側面ごとに短い英語の句を持っています。"
        "実画像を見て側面ごとに確認できたカードだけを展示に出します。</p>"
        + example_rows
        + '<dl class="kv">'
        "<dt>被写体語を含めない</dt><dd>句は表現（色・光・描画・雰囲気）だけを書き、"
        "被写体を表す語を入れません。カードの被写体もお題と重ねていないので、"
        "「被写体が好き」と「表現が好き」を切り分けられます。</dd>"
        "<dt>重みの付け方</dt><dd>句ごとの weight = カードの <code>strength</code>"
        f"（{strengths}）× 側面ごとの倍率 <code>aspect_gain</code>（{gains}）。</dd>"
        "<dt>重複は1件にまとめる</dt><dd>複数のカードで同じ句が選ばれたら、"
        "参照は1件にまとめて weight を合算します。</dd>"
        "<dt>「外す」は除去</dt><dd>外した画像や側面は重み0で残すのではなく、参照リストから取り除きます。"
        "選ばなかった画像を負の重みに変えることもしません。</dd>"
        "</dl></section>"
    )
    # ---------------------------------------------------------------- 4
    others = ", ".join(f"<code>{e(name)}</code>" for name in f["other_policies"])
    profiling = (
        "参照をすべて使う（選別しない）"
        if sample_size == 0
        else f"公式の profiling で参照を選別（<code>{e(sample_size)}</code>）"
    )
    parts.append(
        '<section id="settings"><h2><span class="n">4</span>検証で決めた設定</h2>'
        "<p>エンコーダーの設定は <code>configs/fan-policies.json</code> だけが持ち、"
        f"既定は <code>{e(f['default_id'])}</code> です。"
        "サーバーが解決し、画面やリクエストからは変えられません。</p>"
        '<dl class="kv">'
        f"<dt><code>skip_pa</code> = {e(skip_pa)}</dt>"
        f"<dd>入力側の {len(policy['skip_pa'])} 層では personalized attention を行いません"
        "（CLIP-L・bigG の両方に同じ番号を適用）。"
        "<small>入力側の層で混ぜると参照由来の全体的な「もや」がかかり、"
        "お題の被写体・構図・衣装の指定が崩れやすかったため。</small></dd>"
        f"<dt><code>use_attn_mask</code> = {str(policy['use_attn_mask']).lower()}</dt>"
        "<dd>attention mask を使いません。"
        "<small>true にすると α=0（参照の影響がないはず）でもお題の文のエンコードが変わってしまったため。</small></dd>"
        f"<dt><code>alpha</code> = {alpha}</dt>"
        "<dd>反映の強さ。"
        "<small>当初の確認で α=0.6 までお題の指定が保たれたため、その上限の内側に置いています。"
        "強く反映するほど良いとは限りません。</small></dd>"
        f"<dt><code>pooled_mode</code> = {e(policy['pooled_mode'])}</dt>"
        "<dd>pooled 埋め込みは参照なし（plain）の値を使います。<strong>上流からの意図的な逸脱</strong>です。"
        "<small>公式の ClassTokenDecoder が padding トークンを終端と誤検出するため。"
        "公式どおりの pooled は別の登録 policy で評価と比較に使っています。</small></dd>"
        f"<dt><code>skip</code> = {e(policy['skip'])}</dt>"
        "<dd>hidden states は最終層の1つ手前から取ります（公式の SDXL 設定と同じ）。</dd>"
        f"<dt>profiling</dt><dd>{profiling}。</dd>"
        "</dl>"
        + (
            f'<p class="note">登録済みの別 policy：{others}。既定は評価の事前基準と本人評価の両方を満たすまで変えません。</p>'
            if others
            else ""
        )
        + '<p class="more">各設定を比較した実測と判定は '
        + link("report", "検証記録（/report/）")
        + " を参照してください。</p></section>"
    )
    # ---------------------------------------------------------------- 5
    vae_row = ""
    if vae:
        vae_row = (
            f'<dt>VAE</dt><dd><a href="{e(MODEL_PAGE.format(vae["model"]))}">'
            f"{e(vae['model'])}</a>（revision <code>{e(vae['revision'][:12])}</code>）に差し替え。"
            "<small>チェックポイント同梱の VAE を fp16 で使うと、白っぽく低コントラストになるため。</small></dd>"
        )
    kwargs = ", ".join(
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in g.get("scheduler_kwargs", {}).items()
    )
    parts.append(
        '<section id="generation"><h2><span class="n">5</span>生成設定</h2>'
        "<p>通常生成とパーソナライズ生成で同じ値を使います。値は <code>configs/demo.json</code> の "
        "<code>generation</code> から、このページの生成時に読み込んでいます。</p>"
        '<dl class="kv">'
        f'<dt>生成モデル</dt><dd><a href="{e(MODEL_PAGE.format(g["model"]))}">{e(g["model"])}</a>'
        f"（revision <code>{e(g['revision'][:12])}</code> に固定）"
        f"<small>単一ファイル <code>{e(g['checkpoint'])}</code> を読み込み、"
        "構成ファイルと tokenizer だけを初期 Illustrious の固定 revision から読みます。</small></dd>"
        f"<dt>解像度</dt><dd>{size}（縦長）</dd>"
        f"<dt>steps</dt><dd>{g['steps']}</dd>"
        f"<dt>sampler</dt><dd>{e(f['sampler'])}"
        f"<small><code>{e(g['scheduler'])}</code>（{e(kwargs)}）</small></dd>"
        f"<dt>CFG</dt><dd>{g['guidance_scale']}</dd>"
        f"<dt>精度</dt><dd>{e(g['precision'])}</dd>"
        + vae_row
        + f"<dt>seed</dt><dd>{', '.join(str(seed) for seed in f['seeds'])}</dd>"
        "<dt>負のプロンプト</dt><dd>固定。常に参照なしでエンコードします。</dd>"
        "</dl>"
        + (
            '<p class="note">事前生成した固定画像（<code>assets/manifest.json</code>）も、'
            "この生成設定と一致しています。</p>"
            if f["manifest_matches"]
            else '<p class="note">注意：<code>assets/manifest.json</code> の生成設定が現在の設定と一致していません。</p>'
        )
        + "</section>"
    )
    # ---------------------------------------------------------------- 6
    parts.append(
        '<section id="system"><h2><span class="n">6</span>システム構成</h2>'
        '<dl class="kv">'
        "<dt>Web</dt><dd>FastAPI を1プロセスで起動します。モデルは Web プロセスに読み込みません。"
        "体験は同時に1つだけです。</dd>"
        "<dt>GPU ワーカー</dt><dd>生成ごとに別プロセスを起動し、FAN 専用の環境 "
        f"<code>{e(f['fan']['python'])}</code> で動かします。"
        "<small>FAN の attention の差し替えは transformers 5 系と互換がないため、環境を分けています。"
        "GPU はファイルロック1つで排他し、ワーカーが終了してから解放します。</small></dd>"
        "<dt>通常生成</dt><dd>お題 × seed の画像を事前に生成しておき"
        f"（{len(f['topics'])}お題 × {seeds} = {f['generic_count']}枚）、"
        "その場ではパーソナライズ生成の4枚だけを描きます。</dd>"
        "<dt>run のモード</dt><dd><code>live</code>：その場で生成 / "
        "<code>exact-cache</code>：同じ体験中に同じ内容で生成済みなら、その画像を再表示 / "
        "<code>sample</code>：代表的な選択から事前生成したサンプル（来場者の選択の結果ではありません）</dd>"
        f"<dt>時間の制限</dt><dd>{f['idle']}秒操作がなければ体験を終了し、一時データを削除します"
        f"（生成中は止めます）。1回の生成（{seeds}枚）は {f['timeout']}秒で打ち切ります。</dd>"
        "<dt>中止</dt><dd>比較が成立しないので、その run ごと破棄します。</dd>"
        "<dt>オフライン</dt><dd>ワーカーは Hugging Face / Transformers の offline mode を強制します。"
        "モデルと固定画像が準備済みなら、インターネット接続は要りません。</dd>"
        "</dl></section>"
    )
    # ---------------------------------------------------------------- 7
    parts.append(
        '<section id="limits"><h2><span class="n">7</span>言わないこと</h2>'
        "<p>説明を正確に保つため、次のことは主張しません。</p>"
        '<ul class="dont">'
        "<li><b>attention から因果を説明しない。</b>"
        "attention の値から「この色はこの画像から来た」とは言いません。</li>"
        "<li><b>比較表示は性能の主張ではない。</b>"
        "通常生成とパーソナライズ生成を並べるのは違いを見てもらうためで、"
        "どちらが優れているかを示すものではありません。強く反映するほど良いとも言いません。</li>"
        "<li><b>論文の定量結果を展示の性能として扱わない。</b>"
        f"公式のエンコーダー処理を {model} と展示のカードで動かしていますが、"
        "論文の数値を再現したとは言いません。</li>"
        "</ul></section>"
    )
    parts.append(
        "</main><footer><span>このページは <code>exhibit/scripts/build_tech.py</code> が"
        '設定ファイルから生成しています。</span><span class="links">'
        + link("home", "体験に戻る")
        + link("fallback", "事前生成サンプル")
        + link("report", "検証記録")
        + "</span></footer>"
        + FILE_LINKS
        + "</body></html>\n"
    )
    return "".join(parts)


def main(output=OUTPUT):
    Path(output).write_text(render())


if __name__ == "__main__":
    main()
