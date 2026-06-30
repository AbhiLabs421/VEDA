"""Render UI mockups as SVG (zero deps).  Used for docs / mentor."""

import os
from xml.sax.saxutils import escape


def svg_doc(w, h):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
           f'width="{w}" height="{h}" '
           f'font-family="Inter,Segoe UI,Arial,sans-serif">']
    out.append('''
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#faf9f6"/>
    <stop offset="1" stop-color="#eef1ea"/>
  </linearGradient>
</defs>''')
    out.append(f'<rect width="{w}" height="{h}" fill="url(#bg)"/>')
    return out


def card(out, x, y, w, h, title, body_lines=()):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
               f'fill="#ffffff" stroke="#e4e1d8"/>')
    out.append(f'<text x="{x+18}" y="{y+24}" font-size="11" '
               f'fill="#6b6f64" font-weight="700" letter-spacing="1.2">'
               f'{escape(title.upper())}</text>')
    yy = y + 50
    for ln in body_lines:
        out.append(f'<text x="{x+18}" y="{yy}" font-size="13" fill="#21241f">'
                   f'{escape(ln)}</text>')
        yy += 22


def header(out, role, label):
    out.append('<rect width="1100" height="56" fill="#ffffff" '
               'stroke="#e4e1d8" stroke-width="0 0 1 0"/>')
    out.append('<text x="24" y="34" font-size="18" font-weight="700" '
               'fill="#21241f">VedaX KM Agent</text>')
    out.append('<text x="850" y="34" font-size="13" fill="#6b6f64">'
               f'{label}</text>')
    color = {"superuser": "#7c3aed", "admin": "#1e5a96",
             "user": "#4a5d4e"}[role]
    out.append(f'<rect x="970" y="18" width="86" height="20" rx="10" '
               f'fill="{color}"/>')
    out.append(f'<text x="1013" y="32" font-size="10" font-weight="700" '
               f'fill="#fff" text-anchor="middle" letter-spacing="1.2">'
               f'{role.upper()}</text>')


def tabs(out, role):
    all_tabs = [("Ask", True),
                ("Documents", role in ("admin", "superuser")),
                ("Audit &amp; Trends", role in ("admin", "superuser")),
                ("Pending approvals", role == "superuser"),
                ("Users", role == "superuser")]
    out.append('<rect x="0" y="56" width="1100" height="42" fill="#ffffff" '
               'stroke="#e4e1d8" stroke-width="0 0 1 0"/>')
    x = 24
    first = True
    for label, visible in all_tabs:
        if not visible:
            continue
        color = "#21241f" if first else "#6b6f64"
        out.append(f'<text x="{x}" y="84" font-size="14" fill="{color}" '
                   f'font-weight="{"600" if first else "400"}">{label}</text>')
        if first:
            text_w = 14 + len(label) * 8
            out.append(f'<rect x="{x-12}" y="96" width="{text_w}" '
                       f'height="2" fill="#4a5d4e"/>')
            first = False
        x += len(label) * 8 + 26


# ─────────── 1. user view: only Ask
def user_view():
    out = svg_doc(1100, 580)
    header(out, "user", "alice@xyz.in")
    tabs(out, "user")
    card(out, 24, 122, 1052, 240,
         "Ask the SOP knowledge base", [
             "🔍  casual leave kitne din milti hai",
             "",
             "🟢 Answered.",
             "Every confirmed employee is entitled to 12 days of Casual",
             "Leave per calendar year. [1]",
             "",
             "Grounded 100%   ·   Source: hr_policy.txt"
         ])
    out.append('<text x="24" y="400" font-size="12" fill="#6b6f64">'
               '↑ user can only see the Ask tab. No documents / users / '
               'audit menus shown.</text>')
    return "".join(out) + "</svg>"


# ─────────── 2. admin view: ask + docs + audit
def admin_view():
    out = svg_doc(1100, 580)
    header(out, "admin", "bob@xyz.in")
    tabs(out, "admin")
    card(out, 24, 122, 1052, 100,
         "Upload SOP document", [
             "[ choose file… ]      [ category ]      [ tags ]      [Upload]"
         ])
    card(out, 24, 240, 1052, 250,
         "Indexed documents", [
             "✓  hr_policy.pdf           HR  v1.0      auto-fetched     [Delete]",
             "✓  retail_direct.pdf       Compliance  v1.0   [Delete]",
             "✓  finance_sop.txt         Finance  v1.0      [Delete]",
             "",
             "[ Rescan ./sop_docs folder ]"
         ])
    return "".join(out) + "</svg>"


# ─────────── 3. superuser view: user management
def superuser_view():
    out = svg_doc(1100, 580)
    header(out, "superuser", "carol@xyz.in (you)")
    tabs(out, "superuser")
    card(out, 24, 122, 1052, 380, "Users  ·  superuser only", [
        "Username       Name        Role          Status      Last login           Actions",
        "────────────────────────────────────────────────────────────────────────",
        "alice          Alice U     user          active      2 min ago            [Make admin] [Revoke] [Delete]",
        "bob            Bob A       admin         active      5 min ago            [Make user] [Make superuser] [Revoke] [Delete]",
        "daniel         Daniel R    user          REVOKED     1 hr ago             [Restore] [Make admin] [Delete]",
        "carol          Carol R     superuser     active      now                  (you)",
    ])
    return "".join(out) + "</svg>"


# ─────────── 4. login page
def login_view():
    out = svg_doc(1100, 580)
    out.append('<rect width="1100" height="580" fill="#faf9f6"/>')
    cx = 550
    out.append(f'<rect x="{cx-180}" y="170" width="360" height="280" rx="14" '
               f'fill="#ffffff" stroke="#e4e1d8"/>')
    out.append(f'<text x="{cx}" y="218" font-size="20" font-weight="700" '
               f'fill="#21241f" text-anchor="middle">VedaX KM Agent</text>')
    out.append(f'<text x="{cx}" y="244" font-size="13" fill="#6b6f64" '
               f'text-anchor="middle">Sign in with LDAP / Keycloak</text>')
    out.append(f'<text x="{cx-150}" y="280" font-size="11" fill="#6b6f64">USERNAME</text>')
    out.append(f'<rect x="{cx-150}" y="288" width="300" height="36" rx="6" '
               f'fill="#faf9f6" stroke="#e4e1d8"/>')
    out.append(f'<text x="{cx-150}" y="345" font-size="11" fill="#6b6f64">PASSWORD</text>')
    out.append(f'<rect x="{cx-150}" y="353" width="300" height="36" rx="6" '
               f'fill="#faf9f6" stroke="#e4e1d8"/>')
    out.append(f'<rect x="{cx-150}" y="405" width="300" height="38" rx="8" '
               f'fill="#4a5d4e"/>')
    out.append(f'<text x="{cx}" y="430" font-size="14" font-weight="600" '
               f'fill="#ffffff" text-anchor="middle">Sign in</text>')
    return "".join(out) + "</svg>"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "docs", "ui-mockups")
    os.makedirs(out, exist_ok=True)
    for name, fn in (("login", login_view),
                     ("user_view", user_view),
                     ("admin_view", admin_view),
                     ("superuser_view", superuser_view)):
        with open(os.path.join(out, f"{name}.svg"), "w") as f:
            f.write(fn())
    print(f"wrote 4 UI mockups in {out}")


if __name__ == "__main__":
    main()
