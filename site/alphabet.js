(() => {
'use strict';
const $ = id => document.getElementById(id), KEY = 'typology-alphabet-v1';
let letters = [], selection = null, selected = null, sourceImage = null, start = null, offset = 0, query = '', request = 0;
const canvas = $('editor'), ctx = canvas.getContext('2d');
const safeURL = value => { try { const u = new URL(value, location.href); return ['http:', 'https:'].includes(u.protocol) ? u.href : ''; } catch { return ''; } };
function valid(item) { return item && typeof item.letter === 'string' && [...item.letter].length === 1 && typeof item.image === 'string' && /^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(item.image) && item.image.length < 500000 && typeof item.id === 'string'; }
try { const saved = JSON.parse(localStorage.getItem(KEY) || '[]'); if (Array.isArray(saved)) letters = saved.filter(valid).slice(0, 500); } catch {}
function persist() { try { localStorage.setItem(KEY, JSON.stringify(letters)); return true; } catch { $('save-status').textContent = 'Browser storage is full. Download your alphabet to keep these letters.'; return false; } }
function render() {
  const poster = $('poster'); poster.replaceChildren(); $('print-sources').replaceChildren();
  let collected = 0, total = 0; const used = new Map();
  for (const word of $('sentence').value.toUpperCase().split(/\s+/)) {
    const span = document.createElement('span'); span.className = 'word';
    for (const letter of word) {
      total++; const choices = letters.filter(l => l.letter.toUpperCase() === letter), item = choices[Math.floor(Math.random() * choices.length)];
      if (!item) { const el = document.createElement('span'); el.className = 'placeholder'; el.textContent = letter; el.title = `Collect ${letter} to replace this placeholder`; span.append(el); continue; }
      collected++; const link = document.createElement('a'); link.className = 'glyph'; link.href = safeURL(item.source) || `index.html#layers=store_sign,street_name_plate&q=${encodeURIComponent(item.text || '')}`; link.target = '_blank'; link.rel = 'noopener'; link.title = `${letter} · ${item.text || 'Sign'}${item.estimated ? ' · OCR character, check before using' : ''}`;
      const img = new Image(); img.src = item.image; img.alt = letter; link.append(img); span.append(link); used.set(item.id, item);
    } poster.append(span);
  }
  $('coverage').textContent = `${collected} of ${total} letters from signs · ${letters.length} saved variants${letters.some(l => l.estimated) ? ' · Starter letters use OCR character alignment. Inspect and trim any incorrect letters.' : ''}`;
  for (const item of used.values()) { const li = document.createElement('li'); li.textContent = `${item.text || item.letter} · ${item.source || item.id}`; $('print-sources').append(li); }
}
function renderCollection() {
  $('collection').replaceChildren();
  for (const item of letters) { const card = document.createElement('div'); card.className = 'saved'; const img = new Image(); img.src = item.image; img.alt = item.letter; const label = document.createElement('span'); label.textContent = item.letter + (item.estimated ? ' ≈' : ''); const remove = document.createElement('button'); remove.textContent = 'Remove'; remove.setAttribute('aria-label', `Remove ${item.letter} variant`); remove.onclick = () => { letters.splice(letters.indexOf(item), 1); persist(); renderCollection(); render(); }; card.append(img, label); if (item.crop) { const edit = document.createElement('button'); edit.textContent = 'Trim'; edit.setAttribute('aria-label', `Trim ${item.letter} variant`); edit.onclick = () => { $('letter').value = item.letter; choose({...item, initialRect:item.rect}); $('editor').scrollIntoView({behavior:'smooth',block:'center'}); }; card.append(edit); } card.append(remove); $('collection').append(card); }
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height); if (!sourceImage) return;
  ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);
  if (selection) { const [x,y,w,h] = selection; ctx.fillStyle = '#0007'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.drawImage(sourceImage,x*sourceImage.width,y*sourceImage.height,w*sourceImage.width,h*sourceImage.height,x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height); ctx.strokeStyle = '#ffbf47'; ctx.lineWidth = 2; ctx.strokeRect(x*canvas.width,y*canvas.height,w*canvas.width,h*canvas.height); }
}
function fields() { ['left','top','width','height'].forEach((id,i) => { $(id).value = Math.round(selection[i]*100); }); }
function point(e) { const r = canvas.getBoundingClientRect(); return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)), Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]; }
canvas.onpointerdown = e => { if (!sourceImage) return; start = point(e); canvas.setPointerCapture(e.pointerId); };
canvas.onpointermove = e => { if (!start) return; const p = point(e); selection = [Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.abs(start[0]-p[0]),Math.abs(start[1]-p[1])]; fields(); draw(); };
canvas.onpointerup = canvas.onpointercancel = () => { start = null; };
for (const id of ['left','top','width','height']) $(id).oninput = () => { const n = ['left','top','width','height'].map(k => Number($(k).value)/100); if (!n.every(Number.isFinite)) return; selection = [Math.max(0,Math.min(.99,n[0])),Math.max(0,Math.min(.99,n[1])),Math.max(.01,Math.min(1,n[2])),Math.max(.01,Math.min(1,n[3]))]; selection[2]=Math.min(selection[2],1-selection[0]); selection[3]=Math.min(selection[3],1-selection[1]); draw(); };
async function choose(row) {
  selected = row; sourceImage = null; $('save').disabled = true; $('source').hidden = true; $('save-status').textContent = 'Loading sign…';
  const img = new Image(); img.onload = () => { if (selected !== row) return; sourceImage = img; canvas.width = Math.min(600,img.width); canvas.height = Math.round(canvas.width*img.height/img.width); selection = row.initialRect || [0,0,1,1]; fields(); draw(); $('save').disabled = false; $('save-status').textContent = 'Drag around one letter, enter it below, then save.'; };
  img.onerror = () => { $('save-status').textContent = 'This crop could not be loaded. Choose another sign.'; };
  // Same-origin crops keep canvas exports available without depending on remote CORS.
  img.src = row.crop;
  try { const detail = await fetch('/api/object?' + new URLSearchParams({id:row.id,layers:'store_sign,street_name_plate',confidence:'.65'})).then(r => { if (!r.ok) throw Error(); return r.json(); }); if (selected !== row) return; row.source = safeURL(detail.u); $('source').href = row.source; $('source').hidden = !row.source; } catch {}
}
async function search(more=false) {
  const generation = ++request; if (!more) { offset = 0; query = $('search').value.trim(); $('results').replaceChildren(); }
  $('more').disabled = true; $('search-status').textContent = 'Finding signs…';
  try { const res = await fetch('/api/query?' + new URLSearchParams({layers:'store_sign,street_name_plate',confidence:'.65',q:query,limit:'24',offset:String(offset),map:'0'})); if (!res.ok) throw Error(); const data = await res.json(); if (generation !== request) return;
    for (const row of data.items) { if (!row.crop) continue; const button = document.createElement('button'); button.className = 'sign'; const img = new Image(); img.loading = 'lazy'; img.src = row.crop; img.alt = row.text || row.t || 'Sign'; const caption = document.createElement('span'); caption.textContent = row.text || row.t || 'Untranscribed sign'; button.append(img,caption); button.onclick = () => choose(row); $('results').append(button); }
    offset += data.items.length; $('more').hidden = !data.has_more; $('search-status').textContent = data.items.length ? 'Choose a sign to collect a letter.' : 'No signs found. Try another word.';
  } catch { if (generation === request) $('search-status').textContent = 'Could not load signs. Please try again.'; } finally { if (generation === request) $('more').disabled = false; }
}
$('save').onclick = () => {
  const letter = $('letter').value.trim().toUpperCase(); if ([...letter].length !== 1 || !sourceImage || !selection || selection[2]*sourceImage.width < 3 || selection[3]*sourceImage.height < 3) { $('save-status').textContent = 'Select a letter at least 3 pixels wide and tall, and enter one character.'; return; }
  if (letters.length >= 500) { $('save-status').textContent = 'Alphabet limit reached. Remove a variant first.'; return; }
  const [x,y,w,h] = selection, out = document.createElement('canvas'); out.height = Math.min(160,Math.round(h*sourceImage.height)); out.width = Math.min(320,Math.round(out.height*w*sourceImage.width/(h*sourceImage.height))); out.width = Math.max(1,out.width);
  try { out.getContext('2d').drawImage(sourceImage,x*sourceImage.width,y*sourceImage.height,w*sourceImage.width,h*sourceImage.height,0,0,out.width,out.height); letters.push({letter,image:out.toDataURL('image/png'),id:selected.id,text:selected.text || selected.t || '',source:selected.source || '',crop:selected.crop,rect:selection.slice()}); const saved = persist(); renderCollection(); render(); if (saved) $('save-status').textContent = `${letter} saved. Collect another letter or another version.`; } catch { $('save-status').textContent = 'This image cannot be exported. Try another sign.'; }
};
$('sentence').oninput = render; $('shuffle').onclick = render; $('print').onclick = () => window.print(); $('search-form').onsubmit = e => { e.preventDefault(); search(); }; $('more').onclick = () => search(true);
$('backup').onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify({version:1,letters},null,2)],{type:'application/json'})); const a = document.createElement('a'); a.href=url;a.download='coruna-alphabet.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000); };
$('restore').onchange = async e => { try { const file=e.target.files[0]; if (!file) return; if (file.size>12000000) throw Error(); const data=JSON.parse(await file.text()); if(data.version!==1 || !Array.isArray(data.letters) || data.letters.length>500 || !data.letters.every(valid)) throw Error(); letters=data.letters;const stored=persist();renderCollection();render();if(stored)$('save-status').textContent='Alphabet imported.'; } catch { $('save-status').textContent='Could not import this alphabet. Use a downloaded alphabet JSON file.'; } e.target.value=''; };
renderCollection();render();search();
fetch('alphabet-data.json', {cache:'no-cache'}).then(r=>r.ok?r.json():null).then(data=>{
  if (!data || !Array.isArray(data.letters)) return;
  const starter = data.letters.filter(valid).slice(0,500);
  if (!letters.length) { letters=starter;persist();renderCollection();render();return; }
  const key = item => item.letter+'|'+item.id+'|'+item.image;
  const existing = new Set(letters.map(key)), additions = starter.filter(item=>!existing.has(key(item)));
  if (!additions.length) return;
  $('add-starter').hidden=false; $('add-starter').textContent=`Add ${additions.length} new starter letters`;
  $('add-starter').onclick=()=>{
    const current = new Set(letters.map(key));
    const available = Math.max(0,500-letters.length), added=starter.filter(item=>!current.has(key(item))).slice(0,available);
    letters.push(...added);const stored=persist();renderCollection();render();$('add-starter').hidden=true;
    if(stored)$('save-status').textContent=`Added ${added.length} starter letters. Your existing letters are preserved.`;
  };
}).catch(()=>{});
})();
