(() => {
'use strict';
const $ = id => document.getElementById(id), KEY = 'typology-alphabet-v1';
let letters = [], selection = null, selected = null, sourceImage = null, start = null, offset = 0, query = '', request = 0;
const canvas = $('editor'), ctx = canvas.getContext('2d');
const safeURL = value => { try { const u = new URL(value, location.href); return ['http:', 'https:'].includes(u.protocol) ? u.href : ''; } catch { return ''; } };
function valid(item) { return item && typeof item.letter === 'string' && [...item.letter].length === 1 && typeof item.image === 'string' && /^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(item.image) && item.image.length < 500000 && typeof item.id === 'string'; }
try { const saved = JSON.parse(localStorage.getItem(KEY) || '[]'); if (Array.isArray(saved)) letters = saved.filter(valid).slice(0, 500); } catch {}
function persist() { try { localStorage.setItem(KEY, JSON.stringify(letters)); return true; } catch { $('save-status').textContent = 'Browser storage is full. Download your alphabet to keep these letters.'; return false; } }
const contextCache = new Map();
let contextIntent = null, contextTimer = null, contextSerial = 0;
function closeContext() {
  clearTimeout(contextTimer); contextSerial++;
  if (contextIntent) contextIntent.classList.remove('previewing');
  contextIntent = null; $('letter-context').hidden = true;
}
function contextRoute(item) {
  const match = /^\/proposal-assets\/(apple|gsv)\/([a-f0-9]{12})\/([A-Za-z0-9_-]{1,160})\/([A-Za-z0-9_-]{1,160})\.jpg$/.exec(item.crop || '');
  return match ? `/proposal-context/${match[1]}/${match[2]}/${match[3]}/${match[4]}.json` : null;
}
async function revealContext(link, item) {
  closeContext(); contextIntent = link; const serial = contextSerial;
  const route = contextRoute(item), rect = item.rect;
  if (!route || !Array.isArray(rect) || rect.length !== 4 || !rect.every(Number.isFinite) || rect[2] <= 0 || rect[3] <= 0 || rect[0] < 0 || rect[1] < 0 || rect[0]+rect[2] > 1.001 || rect[1]+rect[3] > 1.001) { link.dataset.contextUnavailable='true'; closeContext(); return; }
  try {
    if (!contextCache.has(route)) {
      if (contextCache.size >= 32) contextCache.delete(contextCache.keys().next().value);
      contextCache.set(route, fetch(route).then(r=>{if(!r.ok)throw Error();return r.json();}));
    }
    const geometry = await contextCache.get(route);
    if (serial !== contextSerial || contextIntent !== link) return;
    if (!Array.isArray(geometry.crop_box) || geometry.crop_box.length !== 4 || !geometry.crop_box.every(Number.isFinite) || !/^\/proposal-context\/(apple|gsv)\/[a-f0-9]{12}\/[A-Za-z0-9_-]+\.jpg$/.test(geometry.context_url)) throw Error();
    const [left,top,right,bottom] = geometry.crop_box;
    const bounds = [left+rect[0]*(right-left), top+rect[1]*(bottom-top), rect[2]*(right-left), rect[3]*(bottom-top)];
    if (bounds[2] <= 0 || bounds[3] <= 0) throw Error();
    const image = new Image();
    await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=reject;image.src=geometry.context_url;});
    if (serial !== contextSerial || contextIntent !== link || !link.isConnected) return;
    const tile = link.querySelector('img'), box = tile.getBoundingClientRect();
    const scale = Math.min(box.width/tile.naturalWidth,box.height/tile.naturalHeight);
    const width = tile.naturalWidth*scale, height = tile.naturalHeight*scale;
    const x = box.left+(box.width-width)/2, y = box.top+(box.height-height)/2;
    const photo = $('letter-context-image'); photo.src=image.src;
    const photoScale=Math.min(width/(bounds[2]*image.naturalWidth),height/(bounds[3]*image.naturalHeight));
    const photoWidth=image.naturalWidth*photoScale, photoHeight=image.naturalHeight*photoScale;
    photo.style.width=photoWidth+'px';photo.style.height=photoHeight+'px';
    photo.style.left=x+width/2-(bounds[0]+bounds[2]/2)*photoWidth+'px';
    photo.style.top=y+height/2-(bounds[1]+bounds[3]/2)*photoHeight+'px';
    $('letter-context-caption').textContent=`${item.letter} · ${item.text || 'Sign'} · Source photo © ${item.crop.includes('/gsv/')?'Google':'Apple'} · Escape to close`;
    link.classList.add('previewing'); $('letter-context').hidden=false;
  } catch {
    contextCache.delete(route);
    if (serial === contextSerial) {link.dataset.contextUnavailable='true';closeContext();}
  }
}
function bindContext(link,item) {
  link.addEventListener('pointerenter',e=>{if(e.pointerType==='touch')return;clearTimeout(contextTimer);contextTimer=setTimeout(()=>revealContext(link,item),150);});
  link.addEventListener('pointerleave',e=>{if(e.pointerType==='touch')return;clearTimeout(contextTimer);if(contextIntent===link)closeContext();});
  link.addEventListener('focus',()=>revealContext(link,item));
  link.addEventListener('blur',()=>{if(contextIntent===link)closeContext();});
  let touch=false, touchWasPreview=false;
  link.addEventListener('pointerdown',e=>{touch=e.pointerType==='touch';touchWasPreview=contextIntent===link&&!$('letter-context').hidden;});
  link.addEventListener('click',e=>{if(touch&&!touchWasPreview&&!link.dataset.contextUnavailable){e.preventDefault();if(contextIntent!==link)revealContext(link,item);}else if(touch){closeContext();}touch=false;});
}
$('letter-context-close').onclick=closeContext;
document.addEventListener('pointerdown',e=>{if(contextIntent&&!e.target.closest('.glyph')&&!e.target.closest('#letter-context-close'))closeContext();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeContext();});
window.addEventListener('scroll',closeContext,{passive:true});window.addEventListener('resize',closeContext);
function render() {
  closeContext();
  const poster = $('poster'); poster.replaceChildren(); $('print-sources').replaceChildren();
  let collected = 0, total = 0; const used = new Map();
  for (const word of $('sentence').value.toUpperCase().split(/\s+/)) {
    const span = document.createElement('span'); span.className = 'word';
    for (const letter of word) {
      total++; const choices = letters.filter(l => l.letter.toUpperCase() === letter), item = choices[Math.floor(Math.random() * choices.length)];
      if (!item) { const el = document.createElement('span'); el.className = 'placeholder'; el.textContent = letter; el.title = `Collect ${letter} to replace this placeholder`; span.append(el); continue; }
      collected++; const link = document.createElement('a'); link.className = 'glyph'; link.href = safeURL(item.source) || `index.html#layers=store_sign,street_name_plate&q=${encodeURIComponent(item.text || '')}`; link.target = '_blank'; link.rel = 'noopener'; link.title = `${letter} · ${item.text || 'Sign'}${item.estimated ? ' · OCR character, check before using' : ''}`;
      const img = new Image(); img.src = item.image; img.alt = letter; link.append(img); bindContext(link,item); span.append(link); used.set(item.id, item);
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
  const starter = data.letters.filter(valid).slice(0,500), revision = String(data.revision || starter.length);
  let seen = null; try { seen = localStorage.getItem(KEY+'-starter'); } catch {}
  if (letters.length && seen === revision) return;
  // A new starter revision replaces the previous OCR letters; letters the visitor saved or trimmed stay.
  const own = letters.filter(item=>!item.estimated), added = starter.slice(0,Math.max(0,500-own.length));
  const before = letters.filter(item=>item.estimated).length;
  letters = [...own,...added];
  if (persist()) { try { localStorage.setItem(KEY+'-starter', revision); } catch {} }
  renderCollection();render();
  if (before && added.length !== before) $('save-status').textContent=`Starter alphabet updated to ${added.length} letters. Your own letters are preserved.`;
}).catch(()=>{});
})();
