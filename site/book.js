(async function(){
  'use strict';
  const $=id=>document.getElementById(id);
  const status=$('status');
  let meta, storage=CropStorage.create(), seed=Math.floor(Math.random()*2147483646)+1, generation=0;
  const labels=new Map();
  function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
  async function json(url){const response=await fetch(url);if(!response.ok)throw new Error(`Request failed (${response.status})`);return response.json();}
  function busy(value){document.querySelectorAll('#settings input,#settings select').forEach(node=>{node.disabled=value;});$('compose').disabled=value;$('shuffle').disabled=value;$('print').disabled=true;}
  function params(){return new URLSearchParams({snapshot:meta.query_snapshot,layers:$('category').value,q:$('query').value.trim(),confidence:$('confidence').value,source:$('source').value,barrio:$('barrio').value,group:'1',view:'wall',map:'0',limit:Math.min(120,Number($('pages').value)*Number($('density').value)),seed});}
  function cropImage(item,anchor){return new Promise(resolve=>{
    const img=el('img');img.alt=item.text||item.t||labels.get(item.category)||item.category;img.decoding='async';
    let done=false;
    const finish=ok=>{if(done)return;done=true;clearTimeout(timer);if(!ok){img.remove();anchor.append(el('span','Image unavailable','missing'));}resolve(ok);};
    const timer=setTimeout(()=>finish(false),30000);
    img.onload=()=>finish(true);anchor.append(img);storage.image(img,item.crop,()=>finish(false));
  });}
  async function compose(){
    const current=++generation;busy(true);status.textContent='Selecting objects…';
    try{
      const selection=params(),data=await json('/api/query?'+selection);
      if(current!==generation)return;
      const perPage=Number($('density').value),pageCount=Math.ceil(data.items.length/perPage);
      const title=$('title').value.trim()||'A Coruña, collected';
      const category=labels.get($('category').value)||$('category').value;
      const subtitle=[category,$('query').value.trim()?`Words: ${$('query').value.trim()}`:'',$('barrio').value?$('barrio').selectedOptions[0].textContent:'A Coruña',`Confidence ≥ ${$('confidence').value}`].filter(Boolean).join(' · ');
      const fragment=document.createDocumentFragment(),images=[];
      for(let page=0;page<pageCount;page++){
        const sheet=el('article',undefined,'sheet'),heading=el('div',undefined,'sheet-heading');
        heading.append(el('h2',title),el('span',`${page+1} / ${pageCount}`,'folio'));sheet.append(heading,el('p',subtitle,'sheet-subtitle'));
        const grid=el('div',undefined,'sheet-grid');grid.style.setProperty('--cols',perPage===6?2:perPage===12?3:4);grid.style.setProperty('--rows',perPage===6?3:perPage===12?4:5);
        for(const [index,item] of data.items.slice(page*perPage,(page+1)*perPage).entries()){
          const figure=el('figure'),anchor=el('a');
          const link=new URLSearchParams(selection);link.delete('snapshot');link.delete('limit');link.delete('map');link.set('object',item.object_id||item.id);link.set('observation',item.id);
          link.set('layers',item.category);link.set('lat',item.la);link.set('lon',item.lo);link.set('z','18');
          anchor.href='index.html#'+link;anchor.target='_blank';anchor.rel='noopener';anchor.title='Open object and original image';
          const caption=el('figcaption');
          caption.append(el('span',`${String(page*perPage+index+1).padStart(3,'0')} · ${item.text||item.t||labels.get(item.category)||item.category}`,'caption-name'));
          const provider=item.id.startsWith('apple:')?'Apple':item.id.startsWith('gsv:')?'Google Street View':'Source in object link';
          caption.append(el('span',[item.barrio||'A Coruña',provider].join(' · ')));
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
  $('shuffle').addEventListener('click',()=>{seed=Math.floor(Math.random()*2147483646)+1;compose();});
  $('print').addEventListener('click',()=>window.print());
  $('settings').addEventListener('input',()=>{$('print').disabled=true;status.textContent='Settings changed. Compose the book to update the preview.';});
  const initial=await Promise.allSettled([json('/api/meta'),json('crop-storage.json'),json('barrios.geojson')]);
  if(initial[0].status!=='fulfilled'){status.textContent='Could not load the collection. Reload to retry.';return;}
  meta=initial[0].value;if(initial[1].status==='fulfilled')storage=CropStorage.create(initial[1].value);
  $('category').replaceChildren();
  for(const category of meta.categories){labels.set(category.id,category.label);const option=el('option',category.label);option.value=category.id;$('category').append(option);}
  if(labels.has('graffiti'))$('category').value='graffiti';$('category').disabled=false;
  if(initial[2].status==='fulfilled'){
    const rows=initial[2].value.features.map(f=>f.properties).sort((a,b)=>String(a.name).localeCompare(String(b.name)));
    for(const row of rows){if(row.id===undefined||!row.name)continue;const option=el('option',row.name);option.value=row.id;$('barrio').append(option);}
  }
  busy(false);await compose();
})();
