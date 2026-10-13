(async function(){
  'use strict';
  const $=id=>document.getElementById(id);
  const status=$('status');
  let meta, storage=CropStorage.create(), seed=Math.floor(Math.random()*2147483646)+1, generation=0;
  const labels=new Map(), captionEdits=new Map();
  let captionViews=[];
  const selectedCategories=()=>[...document.querySelectorAll('[data-category]:checked')].map(node=>node.value);
  function categorySummary(){const names=selectedCategories().map(id=>labels.get(id));$('category-summary').textContent=names.length?names.join(', '):'Choose at least one category';}
  function defaultCaption(item){
    const enabled=new Set([...document.querySelectorAll('[data-caption-field]:checked')].map(node=>node.dataset.captionField));
    const source=item.source||item.id.split(':')[0], parts=[];
    if(enabled.has('text')&&(item.text||item.t))parts.push(item.text||item.t);
    if(enabled.has('category'))parts.push(labels.get(item.category)||item.category);
    if(enabled.has('barrio')&&item.barrio)parts.push(item.barrio);
    if(enabled.has('date'))parts.push(item.d?String(item.d).slice(0,10):'Date unknown');
    if(enabled.has('source'))parts.push(source==='apple'?'Apple':source==='gsv'?'Google Street View':'Source in object link');
    return parts.join(' · ');
  }
  function updateCaptions(){for(const view of captionViews){view.text.textContent=captionEdits.has(view.key)?captionEdits.get(view.key):defaultCaption(view.item);view.reset.hidden=!captionEdits.has(view.key);}}
  function addCaption(item,number){
    const caption=el('figcaption'),key=item.object_id||item.id,text=el('span',captionEdits.has(key)?captionEdits.get(key):defaultCaption(item),'caption-text');
    text.contentEditable='plaintext-only';text.setAttribute('role','textbox');text.setAttribute('aria-label',`Caption for object ${number}`);text.setAttribute('aria-multiline','false');text.spellcheck=true;
    const reset=el('button','Reset','caption-reset');reset.type='button';reset.hidden=!captionEdits.has(key);reset.setAttribute('aria-label',`Reset caption for object ${number}`);
    text.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();text.blur();}});
    text.addEventListener('input',()=>{const value=text.textContent.replace(/[\r\n]+/g,' ').slice(0,120);if(text.textContent!==value)text.textContent=value;captionEdits.set(key,value);reset.hidden=false;});
    reset.addEventListener('click',()=>{captionEdits.delete(key);updateCaptions();});
    caption.append(el('span',String(number).padStart(3,'0')+' · ','caption-number'),text,reset);
    captionViews.push({item,key,text,reset});return caption;
  }
  function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
  async function json(url){const response=await fetch(url);if(!response.ok)throw new Error(`Request failed (${response.status})`);return response.json();}
  function busy(value){document.querySelectorAll('#settings input,#settings select').forEach(node=>{node.disabled=value;});$('compose').disabled=value;$('shuffle').disabled=value;$('print').disabled=true;}
  function params(){return new URLSearchParams({snapshot:meta.query_snapshot,layers:selectedCategories().join(','),q:$('query').value.trim(),confidence:$('confidence').value,source:$('source').value,barrio:$('barrio').value,from:$('from').value||'1900',to:$('to').value||'2100',unknown:$('unknown').checked?'1':'0',group:'1',view:'wall',map:'0',limit:Math.min(120,Number($('pages').value)*Number($('density').value)),seed});}
  function cropImage(item,anchor){return new Promise(resolve=>{
    const img=el('img');img.alt=item.text||item.t||labels.get(item.category)||item.category;img.decoding='async';
    let done=false;
    const finish=ok=>{if(done)return;done=true;clearTimeout(timer);if(!ok){img.remove();anchor.append(el('span','Image unavailable','missing'));}resolve(ok);};
    const timer=setTimeout(()=>finish(false),30000);
    img.onload=()=>finish(true);anchor.append(img);storage.image(img,item.crop,()=>finish(false));
  });}
  async function compose(){
    if(!selectedCategories().length){status.textContent='Choose at least one category.';return;}
    if($('from').value&&$('to').value&&Number($('from').value)>Number($('to').value)){status.textContent='The start year must be before the end year.';return;}
    const current=++generation;busy(true);status.textContent='Selecting objects…';
    try{
      const selection=params(),data=await json('/api/query?'+selection);
      if(current!==generation)return;
      const perPage=Number($('density').value),pageCount=Math.ceil(data.items.length/perPage);
      const title=$('title').value.trim();
      const category=selectedCategories().map(id=>labels.get(id)).join(', ');
      const subtitle=[category,$('query').value.trim()?`Words: ${$('query').value.trim()}`:'',$('barrio').value?$('barrio').selectedOptions[0].textContent:'A Coruña',`Confidence ≥ ${$('confidence').value}`,($('from').value||$('to').value)?`${$('from').value||'Any year'}–${$('to').value||'present'}${$('unknown').checked?' + undated':''}`:!$('unknown').checked?'Known dates only':''].filter(Boolean).join(' · ');
      const fragment=document.createDocumentFragment(),images=[];captionViews=[];
      for(let page=0;page<pageCount;page++){
        const sheet=el('article',undefined,'sheet'),heading=el('div',undefined,'sheet-heading');
        const name=el('h2',title);name.hidden=!title;heading.append(name,el('span',`${page+1} / ${pageCount}`,'folio'));sheet.append(heading,el('p',subtitle,'sheet-subtitle'));
        const grid=el('div',undefined,'sheet-grid');grid.style.setProperty('--cols',perPage===6?2:perPage===12?3:4);grid.style.setProperty('--rows',perPage===6?3:perPage===12?4:5);
        for(const [index,item] of data.items.slice(page*perPage,(page+1)*perPage).entries()){
          const figure=el('figure'),anchor=el('a');
          const link=new URLSearchParams(selection);link.delete('snapshot');link.delete('limit');link.delete('map');link.set('object',item.object_id||item.id);link.set('observation',item.id);
          link.set('layers',item.category);link.set('lat',item.la);link.set('lon',item.lo);link.set('z','18');
          anchor.href='index.html#'+link;anchor.target='_blank';anchor.rel='noopener';anchor.title='Open object and original image';
          const caption=addCaption(item,page*perPage+index+1);
          figure.append(anchor,caption);grid.append(figure);images.push(cropImage(item,anchor));
        }
        sheet.append(grid);
        const footer=el('footer',undefined,'sheet-footer');footer.append(el('span','Typology Machine · Imagery © respective providers. Artwork belongs to its authors.'),el('span',`Selection ${seed}`));sheet.append(footer);fragment.append(sheet);
      }
      $('book').replaceChildren(fragment);
      if(!data.items.length){status.textContent='No objects match. Try a different category, search or confidence.';return;}
      status.textContent=`Loading ${data.items.length} images…`;
      const results=await Promise.all(images);if(current!==generation)return;
      const failed=results.filter(ok=>!ok).length;
      status.textContent=`${data.items.length} objects · ${pageCount} pages · ${Number(data.total).toLocaleString()} matching objects. ${failed?`${failed} images unavailable; marked in the preview.`:'Ready to print.'}`;
      $('print').disabled=false;
    }catch(error){if(current===generation){$('book').replaceChildren();status.textContent='Could not compose the book. Try again. '+error.message;}}
    finally{if(current===generation){document.querySelectorAll('#settings input,#settings select').forEach(node=>{node.disabled=false;});$('compose').disabled=false;$('shuffle').disabled=false;}}
  }
  $('settings').addEventListener('submit',event=>{event.preventDefault();compose();});
  $('shuffle').addEventListener('click',()=>{if(!$('settings').reportValidity())return;seed=Math.floor(Math.random()*2147483646)+1;compose();});
  $('print').addEventListener('click',()=>window.print());
  $('settings').addEventListener('input',event=>{
    if(event.target.hasAttribute('data-caption-field')){updateCaptions();return;}
    if(event.target.id==='title'){const title=$('title').value.trim();document.querySelectorAll('.sheet h2').forEach(node=>{node.textContent=title;node.hidden=!title;});return;}
    categorySummary();$('print').disabled=true;status.textContent='Settings changed. Compose the book to update the preview.';
  });
  $('reset-captions').addEventListener('click',()=>{captionEdits.clear();updateCaptions();});
  const initial=await Promise.allSettled([json('/api/meta'),json('crop-storage.json'),json('barrios.geojson')]);
  if(initial[0].status!=='fulfilled'){status.textContent='Could not load the collection. Reload to retry.';return;}
  meta=initial[0].value;if(initial[1].status==='fulfilled')storage=CropStorage.create(initial[1].value);
  for(const category of meta.categories){labels.set(category.id,category.label);const label=el('label',undefined,'check'),input=el('input');input.type='checkbox';input.value=category.id;input.dataset.category=category.id;input.checked=category.id==='graffiti';label.append(input,document.createTextNode(category.label));$('category-options').append(label);}
  if(!selectedCategories().length)$('category-options').querySelector('input').checked=true;categorySummary();
  if(initial[2].status==='fulfilled'){
    const rows=initial[2].value.features.map(f=>f.properties).sort((a,b)=>String(a.name).localeCompare(String(b.name)));
    for(const row of rows){if(row.id===undefined||!row.name)continue;const option=el('option',row.name);option.value=row.id;$('barrio').append(option);}
  }
  busy(false);await compose();
})();
