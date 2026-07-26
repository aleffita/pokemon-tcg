"""Generate a self-contained HTML deck builder/viewer tool.

Reads EN_Card_Data.csv and card_pages.json, generates an HTML file that:
- Shows all cards in a searchable/filterable grid
- Lets you load a deck.csv and see it visually
- Lets you edit the deck (add/remove cards)
- Exports deck as CSV

Usage:
  python scripts/deck_builder/build_deck_tool.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(ROOT, "EN_Card_Data.csv")
PAGES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_pages.json")
OUT_HTML = os.path.join(ROOT, "scripts", "deck_builder", "deck_builder.html")


def main():
    # Load card data
    cards = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["Card ID"])
            name = row["Card Name"]
            stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").strip()
            hp = row.get("HP", "").strip()
            ptype = row.get("Type", "").strip()
            cards[cid] = {
                "id": cid,
                "name": name,
                "stage": stage,
                "hp": hp,
                "type": ptype,
            }

    # Load image page mapping (for verification)
    with open(PAGES_JSON) as f:
        card_pages = json.load(f)

    # Build card data JSON for the HTML
    cards_json = json.dumps(cards)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PTCG Deck Builder</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }}
.header {{ background: #16213e; padding: 16px 24px; display: flex; align-items: center; gap: 16px; border-bottom: 2px solid #0f3460; }}
.header h1 {{ font-size: 20px; color: #e94560; }}
.header .stats {{ color: #aaa; font-size: 14px; }}
.toolbar {{ background: #16213e; padding: 12px 24px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
.toolbar input, .toolbar select {{ background: #0f3460; border: 1px solid #533483; color: #eee; padding: 8px 12px; border-radius: 6px; font-size: 14px; }}
.toolbar input[type="text"] {{ width: 300px; }}
.toolbar button {{ background: #e94560; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; }}
.toolbar button:hover {{ background: #c73650; }}
.toolbar button.secondary {{ background: #533483; }}
.toolbar button.secondary:hover {{ background: #6a4c9c; }}
.main {{ display: flex; height: calc(100vh - 110px); }}
.deck-panel {{ width: 320px; background: #16213e; border-right: 2px solid #0f3460; overflow-y: auto; padding: 12px; }}
.deck-panel h2 {{ font-size: 16px; margin-bottom: 8px; color: #e94560; }}
.deck-list {{ list-style: none; }}
.deck-list li {{ display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.deck-list li:hover {{ background: #0f3460; }}
.deck-list li .count {{ background: #e94560; color: white; border-radius: 4px; padding: 2px 6px; font-weight: 700; min-width: 20px; text-align: center; }}
.deck-list li .name {{ flex: 1; }}
.deck-list li .type {{ color: #888; font-size: 11px; }}
.cards-panel {{ flex: 1; overflow-y: auto; padding: 12px; }}
.cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
.card {{ background: #16213e; border: 2px solid #0f3460; border-radius: 8px; overflow: hidden; cursor: pointer; transition: all 0.15s; }}
.card:hover {{ border-color: #e94560; transform: translateY(-2px); }}
.card.in-deck {{ border-color: #533483; }}
.card img {{ width: 100%; display: block; }}
.card .info {{ padding: 8px; }}
.card .info .name {{ font-size: 12px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.card .info .meta {{ font-size: 11px; color: #888; }}
.toast {{ position: fixed; bottom: 20px; right: 20px; background: #e94560; color: white; padding: 12px 20px; border-radius: 8px; font-weight: 600; display: none; z-index: 100; }}
textarea {{ background: #0f3460; border: 1px solid #533483; color: #eee; padding: 8px; border-radius: 6px; font-family: monospace; font-size: 12px; width: 100%; height: 80px; resize: vertical; }}
</style>
</head>
<body>
<div class="header">
  <h1>PTCG Deck Builder</h1>
  <div class="stats" id="stats">0/60 cards</div>
</div>
<div class="toolbar">
  <input type="text" id="search" placeholder="Search cards..." oninput="filterCards()">
  <select id="filterType" onchange="filterCards()">
    <option value="">All Types</option>
    <option value="Pokémon">Pokémon</option>
    <option value="Trainer">Trainer</option>
    <option value="Energy">Energy</option>
  </select>
  <button onclick="loadDeckCSV()">Load CSV</button>
  <button class="secondary" onclick="exportDeck()">Export CSV</button>
  <button class="secondary" onclick="clearDeck()">Clear</button>
  <button class="secondary" onclick="pasteDeck()">Paste IDs</button>
  <button id="btnViewDeck" class="secondary" onclick="toggleDeckView()">View Deck</button>
</div>
<div class="main">
  <div class="deck-panel">
    <h2>Deck (<span id="deckCount">0</span>/60)</h2>
    <ul class="deck-list" id="deckList"></ul>
    <div style="margin-top:12px">
      <textarea id="pasteArea" placeholder="Paste card IDs here (one per line or comma-separated)..."></textarea>
    </div>
  </div>
  <div class="cards-panel">
    <div class="cards-grid" id="cardsGrid"></div>
  </div>
</div>
<div class="toast" id="toast"></div>
<input type="file" id="fileInput" accept=".csv,.txt" style="display:none" onchange="handleFile(event)">

<script>
const CARDS = {cards_json};
const deck = {{}}; // cardId -> count

function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2000);
}}

function updateStats() {{
  const total = Object.values(deck).reduce((a, b) => a + b, 0);
  document.getElementById('stats').textContent = `${{total}}/60 cards`;
  document.getElementById('deckCount').textContent = total;
  renderDeckList();
}}

function renderDeckList() {{
  const ul = document.getElementById('deckList');
  ul.innerHTML = '';
  const sorted = Object.entries(deck).sort((a, b) => {{
    const ca = CARDS[a[0]], cb = CARDS[b[0]];
    if (!ca || !cb) return 0;
    return ca.name.localeCompare(cb.name);
  }});
  for (const [id, count] of sorted) {{
    const c = CARDS[id];
    if (!c) continue;
    const li = document.createElement('li');
    li.innerHTML = `<span class="count">${{count}}</span><span class="name">${{c.name}}</span><span class="type">${{c.stage}}</span>`;
    li.onclick = () => removeCard(parseInt(id));
    ul.appendChild(li);
  }}
  // Highlight cards in grid
  document.querySelectorAll('.card').forEach(el => {{
    const cid = parseInt(el.dataset.id);
    el.classList.toggle('in-deck', deck[cid] > 0);
  }});
}}

function addCard(id) {{
  const total = Object.values(deck).reduce((a, b) => a + b, 0);
  if (total >= 60) {{ showToast('Deck is full (60 cards)!'); return; }}
  deck[id] = (deck[id] || 0) + 1;
  updateStats();
}}

function removeCard(id) {{
  if (!deck[id]) return;
  deck[id]--;
  if (deck[id] <= 0) delete deck[id];
  updateStats();
}}

function renderCards(filtered) {{
  const grid = document.getElementById('cardsGrid');
  grid.innerHTML = '';
  for (const c of filtered) {{
    const div = document.createElement('div');
    div.className = 'card' + (deck[c.id] ? ' in-deck' : '');
    div.dataset.id = c.id;
    const count = deck[c.id] || 0;
    div.innerHTML = `
      <img src="card_images/${{c.id}}.jpg" alt="${{c.name}}" onerror="this.style.display='none'">
      <div class="info">
        <div class="name">${{count > 0 ? count + 'x ' : ''}}${{c.id}}: ${{c.name}}</div>
        <div class="meta">${{c.stage}} ${{c.hp ? c.hp + 'HP' : ''}} ${{c.type}}</div>
      </div>`;
    div.onclick = () => addCard(c.id);
    grid.appendChild(div);
  }}
}}

let deckViewMode = false;

function toggleDeckView() {{
  deckViewMode = !deckViewMode;
  const btn = document.getElementById('btnViewDeck');
  btn.textContent = deckViewMode ? 'All Cards' : 'View Deck';
  btn.style.background = deckViewMode ? '#e94560' : '';
  filterCards();
}}

function filterCards() {{
  const q = document.getElementById('search').value.toLowerCase();
  const t = document.getElementById('filterType').value;
  let pool = Object.values(CARDS);
  // Deck view: only show cards in deck
  if (deckViewMode) {{
    pool = pool.filter(c => deck[c.id] > 0);
  }}
  const filtered = pool.filter(c => {{
    if (q && !c.name.toLowerCase().includes(q) && !c.id.toString().includes(q)) return false;
    if (t === 'Pokémon' && !c.stage.includes('Pokémon')) return false;
    if (t === 'Trainer' && !['Item', 'Supporter', 'Stadium', 'Tool'].some(x => c.stage.includes(x))) return false;
    if (t === 'Energy' && !c.stage.includes('Energy')) return false;
    return true;
  }});
  renderCards(filtered);
}}

function loadDeckCSV() {{ document.getElementById('fileInput').click(); }}

function handleFile(e) {{
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {{
    const lines = ev.target.result.split('\\n').map(l => l.trim().rstrip(',')).filter(l => l);
    clearDeck(false);
    for (const line of lines) {{
      const id = parseInt(line);
      if (CARDS[id]) addCard(id);
    }}
    showToast(`Loaded ${{Object.values(deck).reduce((a,b) => a+b, 0)}} cards`);
  }};
  reader.readAsText(file);
  e.target.value = '';
}}

function pasteDeck() {{
  const text = document.getElementById('pasteArea').value;
  const ids = text.split(/[\\s,]+/).map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  clearDeck(false);
  for (const id of ids) {{
    if (CARDS[id]) addCard(id);
  }}
  showToast(`Loaded ${{Object.values(deck).reduce((a,b) => a+b, 0)}} cards from paste`);
}}

function exportDeck() {{
  const total = Object.values(deck).reduce((a, b) => a + b, 0);
  if (total !== 60) {{ showToast(`Deck has ${{total}} cards, need 60!`); return; }}
  const lines = [];
  for (const [id, count] of Object.entries(deck)) {{
    for (let i = 0; i < count; i++) lines.push(id);
  }}
  const csv = lines.join('\\n') + '\\n';
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'deck.csv';
  a.click();
  showToast('Exported deck.csv!');
}}

function clearDeck(show = true) {{
  for (const key in deck) delete deck[key];
  updateStats();
  if (show) showToast('Deck cleared');
}}

// Init
filterCards();
</script>
</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {OUT_HTML}")
    print(f"Cards: {len(cards)}, Images mapped: {len(card_pages)}")
    print(f"Open in browser: file://{OUT_HTML}")


if __name__ == "__main__":
    main()
