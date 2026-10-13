(async function () {
  let language='en';
  try{language=localStorage.getItem('typology-language')||(navigator.language.startsWith('es')?'es':'en');}catch(error){language=navigator.language.startsWith('es')?'es':'en';}
  const translations={
    'Map':'Mapa','Wall':'Galería','Shuffle':'Mezclar','Method':'Método','Object layers':'Capas','Group duplicate observations':'Agrupar observaciones duplicadas','Search text':'Buscar texto','Street or shop name…':'Calle o comercio…','Show results on map':'Ver resultados en el mapa','Capture year':'Año de captura','Include unknown dates':'Incluir fechas desconocidas','Select all':'Seleccionar todo','Clear all':'Limpiar todo','Group':'Grupo','All signs':'Todos los rótulos','Not clustered':'Sin agrupar','Cluster colors':'Colores por grupo',
    'Shop signs':'Rótulos comerciales','Street names':'Nombres de calles','Street name plates':'Placas de calles','Graffiti':'Grafitis','Traffic signs':'Señales de tráfico','Street lights':'Farolas','Bins':'Papeleras','Benches':'Bancos','Manhole covers':'Tapas de registro','Electrical cabinets':'Armarios eléctricos','Junction boxes':'Cajas de conexiones','Doors':'Puertas','Shutters':'Persianas','People':'Personas','Pets':'Mascotas','pilot':'prueba',
    'full image':'imagen completa','show crop':'ver recorte','← prev':'← anterior','next →':'siguiente →','close ✕':'cerrar ✕','map':'mapa','Objects':'Objetos','Object details':'Detalles del objeto','View':'Vista','Viewpoints of this object':'Vistas de este objeto','Observations at this location':'Observaciones en esta ubicación','Earliest capture year':'Primer año de captura','Latest capture year':'Último año de captura','Change language':'Cambiar idioma',
    'New data available · Update':'Hay datos nuevos · Actualizar','Updating selected layers…':'Actualizando las capas seleccionadas…','Update failed. Your current data is unchanged; try again.':'No se pudo actualizar. Se conservan los datos actuales; inténtalo de nuevo.','Loading street image…':'Cargando imagen de la calle…','Street image could not load. Switch to crop, then try again.':'No se pudo cargar la imagen. Cambia al recorte e inténtalo de nuevo.','Image could not load.':'No se pudo cargar la imagen.','Capture date unknown':'Fecha de captura desconocida','SAM detection · camera location':'Detección SAM · ubicación de la cámara','SAM person detection · camera location':'Persona detectada por SAM · ubicación de la cámara','OCR · automatic transcription':'OCR · transcripción automática','Cluster colors apply to previously clustered shop signs':'Los colores se aplican a los rótulos comerciales agrupados previamente','Object layers are unavailable. Saved shop signs are still available.':'Las capas no están disponibles. Los rótulos guardados siguen disponibles.','Items matching the capture-date filter':'Elementos que coinciden con el filtro de fecha','Total in snapshot; enable layer to filter by capture date':'Total de la colección; activa la capa para filtrar por fecha','loading…':'cargando…','retry':'reintentar','matches':'resultados','items':'elementos','observations':'observaciones','on':'activas','in view · click a dot to inspect':'en pantalla · pulsa un punto para inspeccionar','Captured':'Capturada','Prompt':'Prompt','score':'puntuación','views':'vistas','viewpoint':'vista','viewpoints grouped as one object':'vistas agrupadas como un objeto','Text across viewpoints (automatic OCR):':'Texto de las distintas vistas (OCR automático):','Street image containing':'Imagen de la calle con','Could not load':'No se pudo cargar','Select it to retry.':'Selecciona la capa para reintentar.','Person experiment':'Experimento de personas','street views':'vistas de calles'
  };
  Object.assign(translations, {
    'Share view':'Compartir vista','Barrio':'Barrio','All barrios':'Todos los barrios',
    'Image source':'Fuente de imágenes','Both sources':'Ambas fuentes','Only this map area':'Solo esta zona del mapa',
    'Reset filters':'Restablecer filtros','Download selection':'Descargar selección','Format':'Formato',
    'Download data':'Descargar datos','Manifest':'Ficha de datos','Objects in map area':'Objetos en esta zona',
    'In this area':'En esta zona','Back to area':'Volver a la zona','Hide':'Ocultar','Show':'Mostrar',
    'Show more':'Ver más','No objects match these filters.':'No hay objetos con estos filtros.',
    'No objects in this area. Move the map or change filters.':'No hay objetos en esta zona. Mueve el mapa o cambia los filtros.',
    'At this point':'En este punto','Selected objects':'Objetos seleccionados','Selected observations':'Observaciones seleccionadas',
    'Loading selected layers…':'Cargando las capas seleccionadas…','View link copied':'Enlace copiado',
    'Copy this view link':'Copia el enlace a esta vista','Download failed. Try a smaller selection.':'No se pudo descargar. Prueba una selección más pequeña.',
    'Municipal barrio boundaries unavailable.':'No se pueden cargar los límites de los barrios.',
    'Outside mapped barrios':'Fuera de los barrios delimitados','Streets':'Calles','Satellite':'Satélite',
    'rows':'filas','Preparing download…':'Preparando descarga…','Download ready':'Descarga preparada',
    'Bounds: Concello da Coruña':'Límites: Concello da Coruña','Clear selection':'Quitar selección',
    'Record details':'Detalles del registro','records, with matching viewpoints':'registros, con sus vistas coincidentes',
    'Some selected layers failed to load. Retry before downloading.':'Algunas capas no se cargaron. Reinténtalo antes de descargar.'
  });
  const t=value=>language==='es'?(translations[value]||value):value;
  document.documentElement.lang=language;
  // Translate interface nodes only; detected text and provider metadata stay original.
  const $ = id => document.getElementById(id), wall = $('wall');
  for(const element of document.querySelectorAll('nav, .cluster-controls, .object-layers, .detail .arrows, .map-results')){
    const walker=document.createTreeWalker(element,NodeFilter.SHOW_TEXT);let node;
    while(node=walker.nextNode()){const raw=node.nodeValue,trimmed=raw.trim();if(translations[trimmed])node.nodeValue=raw.replace(trimmed,t(trimmed));}
  }
  for(const element of document.querySelectorAll('[aria-label],[placeholder]'))for(const attr of ['aria-label','placeholder'])if(element.hasAttribute(attr))element.setAttribute(attr,t(element.getAttribute(attr)));
  $('d-map').textContent=t('map');
  $('language').textContent=language==='es'?'EN':'ES';$('language').onclick=()=>{try{localStorage.setItem('typology-language',language==='es'?'en':'es');}catch(error){}location.reload();};
  if (matchMedia('(max-width:600px)').matches) $('object-layers').open=false;
  let map, active=[], order=[], browse=[], shown=0, curSign, curObject, full=false;
  let manifest=null, availableManifest=null, selectedId='', restoring=true, ready=false, refreshing=false, exporting=false;
  let panelItems=[], panelShown=0, pointSelection=null, barrioLayer=null;
  let barrioData=null, barrios=[], legacy={meta:{},signs:[]};
  let cropStorage=CropStorage.create();
  const loaded=new Map(), enabled=new Set(), pending=new Map(), failed=new Set();
  const filterCache=new WeakMap(), barrioCache=new Map();
  const categories=new Map(), clusters=new Map();
  const normalize=value=>String(value||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
  const grouped=()=>$('group-duplicates').checked;
  const label=row=>t(categories.get(row.category)?.label||row.category);
  const color=row=>$('cluster-colors').checked&&row.legacy_record?row.cluster_color||'#777':categories.get(row.category)?.color||'#ddd';
  const sourceOf=row=>row.source||(row.src==='g'?'gsv':'apple');
  const itemId=row=>row.object_id||row.id;
  const detail=$('detail');
  let returnFocus=null, statusTimer;
  function notice(message){$('action-status').textContent=message;clearTimeout(statusTimer);statusTimer=setTimeout(()=>$('action-status').textContent='',5000);}
  async function fetchJSON(url){const response=await fetch(url);if(!response.ok)throw Error(`${url}: ${response.status}`);return response.json();}

  // Polygon membership uses the same recorded position shown on the map. No nearest-name fallback.
  function ringContains(ring,x,y){
    let inside=false;
    for(let i=0,j=ring.length-1;i<ring.length;j=i++){
      const [ax,ay]=ring[j],[bx,by]=ring[i];
      const cross=(x-ax)*(by-ay)-(y-ay)*(bx-ax);
      if(Math.abs(cross)<1e-12&&x>=Math.min(ax,bx)&&x<=Math.max(ax,bx)&&y>=Math.min(ay,by)&&y<=Math.max(ay,by))return true;
      if((ay>y)!==(by>y)&&x<(bx-ax)*(y-ay)/(by-ay)+ax)inside=!inside;
    }
    return inside;
  }
  function barrioFor(lon,lat){
    if(!barrioData)return null;
    const key=`${lon},${lat}`;
    if(barrioCache.has(key))return barrioCache.get(key);
    const match=barrios.find(b=>lon>=b.bounds[0]&&lon<=b.bounds[2]&&lat>=b.bounds[1]&&lat<=b.bounds[3]&&b.polygons.some(rings=>ringContains(rings[0],lon,lat)&&!rings.slice(1).some(ring=>ringContains(ring,lon,lat))));
    const result=match?{id:match.id,name:match.name}:null;
    barrioCache.set(key,result);return result;
  }
  function prepareBarrios(data){
    barrioData=data;
    barrios=data.features.map(feature=>{
      const polygons=feature.geometry.type==='Polygon'?[feature.geometry.coordinates]:feature.geometry.coordinates;
      const bounds=[Infinity,Infinity,-Infinity,-Infinity];
      for(const polygon of polygons)for(const [x,y]of polygon[0]){bounds[0]=Math.min(bounds[0],x);bounds[1]=Math.min(bounds[1],y);bounds[2]=Math.max(bounds[2],x);bounds[3]=Math.max(bounds[3],y);}
      return {id:String(feature.properties.id),name:feature.properties.name,feature,polygons,bounds};
    }).sort((a,b)=>a.name.localeCompare(b.name,'gl'));
    for(const barrio of barrios){const option=document.createElement('option');option.value=barrio.id;option.textContent=barrio.name;$('barrio-filter').append(option);}
    const outside=document.createElement('option');outside.value='outside';outside.textContent=t('Outside mapped barrios');$('barrio-filter').append(outside);
    $('barrio-filter').disabled=false;
    const attribution=document.createElement('a');attribution.href=data.metadata.source;attribution.target='_blank';attribution.rel='noopener';attribution.textContent=t('Bounds: Concello da Coruña');$('barrio-status').replaceChildren(attribution);
  }
  function decorate(row){
    const barrio=barrioFor(row.lo,row.la);
    return {...row,barrio_id:barrio?.id||null,barrio:barrio?.name||null};
  }
  function groupObservations(rows){
    const groups=new Map();
    for(const row of rows){const id=row.duplicate_group||row.id;if(!groups.has(id))groups.set(id,[]);groups.get(id).push(row);}
    return [...groups.entries()].map(([id,views])=>{
      views.sort((a,b)=>Number(b.id===b.group_rep)-Number(a.id===a.group_rep)||(b.score||0)*Math.sqrt((b.w||1)*(b.h||1))-(a.score||0)*Math.sqrt((a.w||1)*(a.h||1))||a.id.localeCompare(b.id));
      const texts=[...new Set(views.map(v=>(v.text||v.t||'').trim()).filter(Boolean))];
      return {...views[0],object_id:id,views,search_text:texts.join('\n'),group_text:texts.join('\n')};
    });
  }
  function captureInterval(value){
    const match=String(value||'').match(/^(\d{4})-(\d{2})(?:-(\d{2})(?:T.*)?)?$/);
    if(!match)return null;
    const [,year,month,day]=match,y=Number(year),m=Number(month);
    if(m<1||m>12)return null;
    const last=new Date(Date.UTC(y,m,0)).getUTCDate();
    if(day&&(Number(day)<1||Number(day)>last))return null;
    return day?[`${year}-${month}-${day}`,`${year}-${month}-${day}`]:[`${year}-${month}-01`,`${year}-${month}-${last}`];
  }
  function currentFilters(){return {
    categories:[...enabled].sort(),query:$('object-search').value.trim(),barrio:$('barrio-filter').value,
    source:$('source-filter').value,from:$('year-from').value,to:$('year-to').value,
    unknown:$('date-unknown').checked,grouped:grouped(),area:$('area-filter').checked,
    cluster:$('cluster-filter').value
  };}
  function selectedRows(rows){
    const filters=currentFilters(),key=JSON.stringify(filters);
    let cached=filterCache.get(rows);
    if(cached?.key===key)return cached.rows;
    const filtered=rows.filter(row=>{
      if(filters.source&&sourceOf(row)!==filters.source)return false;
      if(filters.barrio&&(filters.barrio==='outside'?row.barrio_id!==null:row.barrio_id!==filters.barrio))return false;
      const interval=captureInterval(row.d);
      if(interval?(interval[1]<`${filters.from}-01-01`||interval[0]>`${filters.to}-12-31`):!filters.unknown)return false;
      if(filters.cluster&&row.legacy_record&&(filters.cluster==='unassigned'?!!row.cluster_id:row.cluster_id!==filters.cluster))return false;
      return true;
    });
    // Search across the retained viewpoints of a group, so a text match keeps its street context.
    const items=grouped()?groupObservations(filtered):filtered;
    const query=normalize(filters.query);
    const result=query?items.filter(row=>normalize([row.search_text,row.text,row.t,row.b,row.st,row.barrio].filter(Boolean).join(' ')).includes(query)):items;
    filterCache.set(rows,{key,rows:result});return result;
  }
  function inMap(row){return map.getBounds().contains([row.la,row.lo]);}
  function selection(){return $('area-filter').checked?active.filter(inMap):active;}
  function updateYears(){
    const years=[];
    for(const rows of loaded.values())for(const row of rows){const year=Number(String(row.d||'').slice(0,4));if(year>=1900&&year<=new Date().getFullYear())years.push(year);}
    let min=Number($('year-from').min)||2012,max=Number($('year-to').max)||new Date().getFullYear();
    for(const year of years){min=Math.min(min,year);max=Math.max(max,year);}
    const wasMin=$('year-from').value===$('year-from').min,wasMax=$('year-to').value===$('year-to').max;
    for(const id of ['year-from','year-to']){$(id).min=min;$(id).max=max;}
    if(wasMin)$('year-from').value=min;if(wasMax)$('year-to').value=max;
    yearLabels();
  }
  function yearLabels(){$('year-from-value').textContent=$('year-from').value;$('year-to-value').textContent=$('year-to').value;}
  function updateSelection(){
    active=[...enabled].flatMap(id=>selectedRows(loaded.get(id)||[]));
    for(const c of categories.values()){
      const count=document.querySelector(`[data-category-count="${c.id}"]`),rows=loaded.get(c.id);
      if(count&&!pending.has(c.id)){count.textContent=rows?selectedRows(rows).length.toLocaleString():(c.count||0).toLocaleString();count.title=rows?t('Selected objects'):t('Total in snapshot; enable layer to filter by capture date');}
    }
    pointSelection=null;
    $('layer-summary').textContent=`${t('Object layers')} · ${enabled.size} ${t('on')}`;
    $('cluster-controls').hidden=!enabled.has('store_sign')||!clusters.size;
    refreshVisible();if(map)observations.draw();saveState(true);
  }
  function refreshVisible(){
    if(!map)return;
    order=selection();
    const busy=refreshing||[...enabled].some(id=>pending.has(id)||!loaded.has(id));
    $('count').textContent=`${order.length.toLocaleString()} ${grouped()?t('items'):t('observations')}${busy?' · '+t('loading…'):''}`;
    $('focus-results').disabled=!active.length;
    const unavailable=[...enabled].some(id=>failed.has(id));
    $('download-data').disabled=exporting||busy||unavailable||!order.length;
    $('download-manifest').disabled=exporting||busy||unavailable||!order.length;
    $('download-summary').textContent=unavailable?t('Some selected layers failed to load. Retry before downloading.'):busy?t('Loading selected layers…'):`${order.length.toLocaleString()} ${grouped()?t('Selected objects'):t('Selected observations')}`;
    renderWall();renderPanel();
  }
  function parseLayer(c,data){
    let rows=data.features.filter(f=>!f.properties.duplicate_of).map(f=>{
      const p=f.properties;
      return decorate({...p,category:c.id,la:f.geometry.coordinates[1],lo:f.geometry.coordinates[0],w:p.width,h:p.height,d:p.date,u:p.provider_url,src:p.source==='gsv'?'g':'a',proposal:true,run_id:p.run_id||p.raw?.split('/')[3]||null,snapshot_id:c.url.split('/')[1]});
    });
    if(c.id==='store_sign'){
      const originals=new Map(signs.filter(s=>s.observation_id).map(s=>[s.observation_id,s])),seen=new Set();
      rows=rows.map(row=>{const old=originals.get(row.id);seen.add(row.id);return old?{...old,...row,legacy_record:true}:row;});
      rows.push(...signs.filter(s=>!seen.has(s.id)));
    }
    return rows;
  }
  async function ensureLayer(id){
    const c=categories.get(id);
    if(!c?.url||loaded.has(id)&&!loaded.get(id).fallback)return;
    if(pending.has(id))return pending.get(id);
    const generation=manifest?.snapshot_id;
    const task=fetchJSON(c.url).then(data=>{
      if(generation!==manifest?.snapshot_id)return;
      loaded.set(id,parseLayer(c,data));failed.delete(id);if(!failed.size)$('layer-error').textContent='';updateYears();
    }).catch(()=>{failed.add(id);$('layer-error').textContent=`${t('Could not load')} ${t(c.label)}. ${t('Select it to retry.')}`;}).finally(()=>{pending.delete(id);updateSelection();});
    pending.set(id,task);updateSelection();return task;
  }
  function addLayer(c){
    const row=document.createElement('label');row.className='layer-option';
    const input=document.createElement('input');input.type='checkbox';input.dataset.category=c.id;input.checked=enabled.has(c.id);
    const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=c.color;
    const name=document.createElement('span');name.textContent=t(c.label);
    const count=document.createElement('small');count.dataset.categoryCount=c.id;count.textContent=(c.count||0).toLocaleString();
    row.append(input,swatch,name,count);$('layer-options').append(row);
    input.onchange=async()=>{if(input.checked){enabled.add(c.id);await ensureLayer(c.id);}else enabled.delete(c.id);updateSelection();};
  }
  function drawBarrio(fit=false){
    if(barrioLayer){map.removeLayer(barrioLayer);barrioLayer=null;}
    const barrio=barrios.find(b=>b.id===$('barrio-filter').value);
    if(barrio){barrioLayer=L.geoJSON(barrio.feature,{style:{color:'#e8bc6a',weight:2,fillOpacity:.04},interactive:false}).addTo(map);if(fit)map.fitBounds(barrioLayer.getBounds(),{padding:[35,35],maxZoom:17});}
  }
  function tile(row,items){
    const button=document.createElement('button');button.className='tile';button.dataset.objectId=itemId(row);button.style.borderBottomColor=color(row);
    button.setAttribute('aria-label',[label(row),row.text||row.t,row.barrio].filter(Boolean).join(' · '));button.setAttribute('aria-pressed',String(itemId(row)===selectedId));
    const image=document.createElement('img');image.loading='lazy';cropStorage.image(image,row.crop);image.alt='';button.append(image);
    if(row.views?.length>1){const badge=document.createElement('span');badge.className='view-count';badge.textContent=`${row.views.length} ${t('views')}`;button.append(badge);}
    button.onclick=()=>{browse=items;open(row);};return button;
  }
  function more(){const end=Math.min(order.length,shown+120);for(;shown<end;shown++)wall.append(tile(order[shown],order));}
  function renderWall(){if(document.body.dataset.view!=='wall')return;wall.replaceChildren();shown=0;more();if(!order.length){const empty=document.createElement('p');empty.textContent=t('No objects match these filters.');wall.append(empty);}}
  function renderPanel(){
    if(document.body.dataset.view!=='map')return;
    panelItems=pointSelection||active.filter(inMap);panelShown=0;$('map-thumbnails').replaceChildren();
    $('results-title').textContent=`${t(pointSelection?'At this point':'In this area')} · ${panelItems.length.toLocaleString()}`;
    $('results-clear').hidden=!pointSelection;$('results-empty').textContent=panelItems.length?'':t('No objects in this area. Move the map or change filters.');
    if(!$('map-results').classList.contains('collapsed'))morePanel();
  }
  function morePanel(){const end=Math.min(panelItems.length,panelShown+48);for(;panelShown<end;panelShown++)$('map-thumbnails').append(tile(panelItems[panelShown],panelItems));$('results-more').hidden=panelShown>=panelItems.length;}
  function setView(view,save=true){
    document.body.dataset.view=view;$('v-map').setAttribute('aria-pressed',String(view==='map'));$('v-wall').setAttribute('aria-pressed',String(view==='wall'));
    detail.setAttribute('aria-modal',String(view==='wall'));
    if(view==='map'){map.invalidateSize();renderPanel();}else renderWall();
    if(save)saveState(true);
  }
  function showPic(){
    const row=curSign,on=!!(full&&row.raw),img=$('d-img'),status=$('image-status');
    status.textContent=on?t('Loading street image…'):'';img.parentElement.classList.toggle('is-loading',on);
    img.onload=()=>{status.textContent='';img.parentElement.classList.remove('is-loading');};
    const imageFailure=()=>{img.parentElement.classList.add('is-loading');status.textContent=on?t('Street image could not load. Switch to crop, then try again.'):t('Image could not load.');delete detail.dataset.full;};
    img.style.width=on?'':Math.min(800,innerWidth-32,innerHeight*.52*(row.w/row.h||1))+'px';if(on){img.onerror=imageFailure;img.src=row.raw;}else cropStorage.image(img,row.crop,imageFailure);img.alt=on?`${t('Street image containing')} ${label(row)}`:row.text||row.t||label(row);
    if(on&&row.bx){const[x0,y0,x1,y1]=row.bx;Object.assign($('d-box').style,{left:x0*100+'%',top:y0*100+'%',width:(x1-x0)*100+'%',height:(y1-y0)*100+'%'});detail.dataset.full='1';}else delete detail.dataset.full;
    $('full').setAttribute('aria-pressed',String(on));$('full').textContent=t(on?'show crop':'full image');$('full').hidden=!row.raw;
  }
  function open(object,view,save=true){
    if(!detail.hasAttribute('open'))returnFocus=document.activeElement;
    curObject=object;curSign=view||object;selectedId=itemId(object);const row=curSign;
    const views=object.views||[object],index=browse.indexOf(object),viewIndex=views.findIndex(v=>v.id===row.id);
    $('d-text').textContent=row.text||row.t||row.b||label(row);
    $('d-id').textContent=`${label(row)} · ${Math.max(1,index+1)} / ${Math.max(1,browse.length)}${views.length>1?' · '+t('viewpoint')+' '+(viewIndex+1)+' / '+views.length:''}`;
    $('d-where').textContent=[row.barrio,row.st,row.proposal?t('SAM detection · camera location'):null].filter(Boolean).join(' · ');
    $('d-date').textContent=row.d?`${t('Captured')} ${row.d}`:t('Capture date unknown');
    $('d-seen').textContent=row.proposal?`${t('Prompt')}: ${row.prompt} · ${t('score')} ${Number(row.score).toFixed(2)}`:`${row.seen||1} ${t('views')}`;
    $('d-ocr').textContent=row.text?t('OCR · automatic transcription'):'';$('d-ocr').title=row.ocr_model||'';
    $('d-cluster').textContent=[views.length>1?`${views.length} ${t('viewpoints grouped as one object')}`:null,clusters.get(row.cluster_id)?.name].filter(Boolean).join(' · ');
    $('d-link').hidden=!row.u;$('d-link').href=row.u||'#';$('d-link').textContent=sourceOf(row)==='gsv'?'Street View ↗':'Look Around ↗';
    $('prev').disabled=$('next').disabled=browse.length<2;
    $('d-map').onclick=e=>{e.preventDefault();setView('map');map.setView([row.la,row.lo],18);};
    const target=$('viewpoints');target.replaceChildren();target.hidden=views.length<2;
    for(const item of views){const button=document.createElement('button');button.className='nearby-item';button.setAttribute('aria-label',`${t('viewpoint')} ${views.indexOf(item)+1}`);button.setAttribute('aria-pressed',String(item.id===row.id));button.style.borderColor=color(item);const image=document.createElement('img');cropStorage.image(image,item.crop);image.loading='lazy';image.alt='';button.append(image);button.onclick=()=>open(object,item);target.append(button);}
    $('nearby').replaceChildren();$('nearby').hidden=true;
    $('group-text').hidden=views.length<2||!object.group_text;$('group-text').textContent=object.group_text?t('Text across viewpoints (automatic OCR):')+'\n'+object.group_text:'';
    detail.setAttribute('open','');detail.setAttribute('aria-modal',String(document.body.dataset.view==='wall'));showPic();$('close').focus();
    for(const button of document.querySelectorAll('.tile[data-object-id]'))button.setAttribute('aria-pressed',String(button.dataset.objectId===selectedId));
    observations.draw();if(save)saveState(true);
  }
  function close(save=true){detail.removeAttribute('open');selectedId='';curObject=null;curSign=null;observations.draw();if(returnFocus?.isConnected)returnFocus.focus();if(save)saveState();}
  function step(delta){if(!browse.length||!curObject)return;const index=browse.indexOf(curObject);open(browse[(Math.max(0,index)+delta+browse.length)%browse.length]);}

  function downloadRecord(row){
    const views=row.views||[row];
    return {
      id:itemId(row),observation_id:row.id,group_id:row.duplicate_group||null,
      member_ids:views.map(v=>v.id),observation_count:views.length,category:row.category,
      category_status:row.status||(row.proposal?'unverified_proposal':'legacy_detection'),
      source:sourceOf(row),sources:[...new Set(views.map(sourceOf))],capture_date:row.d||null,
      capture_dates:[...new Set(views.map(v=>v.d).filter(Boolean))],latitude:row.la,longitude:row.lo,
      location_basis:row.location_basis||(row.proposal?'camera':'estimated_object'),
      barrio:row.barrio,barrio_id:row.barrio_id,barrio_basis:'recorded_position_in_municipal_polygon',
      score:row.score??null,prompt:row.prompt||null,text:row.group_text||row.text||row.t||'',
      ocr_model:row.ocr_model||null,view_id:row.view_id||null,run_id:row.run_id||null,
      snapshot_id:row.snapshot_id||null,provider_url:row.u||null,
      crop_url:row.crop?new URL(cropStorage.url(row.crop),location.href).href:null,
      context_url:row.raw?new URL(row.raw,location.href).href:null,
      box_normalized:row.bx||null,
      observations:views.map(v=>({id:v.id,source:sourceOf(v),capture_date:v.d||null,
        latitude:v.la,longitude:v.lo,location_basis:v.location_basis||(v.proposal?'camera':'estimated_object'),
        barrio:v.barrio,barrio_id:v.barrio_id,score:v.score??null,text:v.text||v.t||'',
        run_id:v.run_id||null,view_id:v.view_id||null,provider_url:v.u||null,
        crop_url:v.crop?new URL(cropStorage.url(v.crop),location.href).href:null,context_url:v.raw?new URL(v.raw,location.href).href:null}))
    };
  }
  function downloadMetadata(rows){
    const bounds=map.getBounds();
    return {schema_version:1,exported_at:new Date().toISOString(),city:legacy.meta.city||'A Coruña',
      selection_url:location.href,record_type:grouped()?'object_groups':'observations',record_count:rows.length,
      observation_count:rows.reduce((sum,row)=>sum+(row.views?.length||1),0),filters:currentFilters(),
      bounds:$('area-filter').checked?[bounds.getWest(),bounds.getSouth(),bounds.getEast(),bounds.getNorth()]:null,
      grouping:'Partial duplicate proposals; singleton records may still depict the same object.',
      selection:'Category, source, date and barrio filter observations before grouping. Text searches the retained viewpoints. Map-area filtering uses each displayed representative position and retains its matching viewpoints.',
      position:'Coordinates retain the recorded position basis. Barrio follows this position; most new observations use camera coordinates.',
      images:'URLs reference the existing viewer assets. Image files are not included. Provider rights apply.',
      csv:'Arrays and nested records are JSON strings. Spreadsheet formula prefixes are escaped with an apostrophe.',
      barrios:barrioData?.metadata||null,snapshot:manifest,legacy:legacy.meta};
  }
  function saveBlob(parts,type,name){
    const url=URL.createObjectURL(new Blob(parts,{type})),anchor=document.createElement('a');anchor.href=url;anchor.download=name;document.body.append(anchor);anchor.click();anchor.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
  }
  function csvCell(value){
    let text=value==null?'':typeof value==='object'?JSON.stringify(value):String(value);
    if(typeof value==='string'&&/^[\s]*[=+@-]/.test(text))text="'"+text;
    return '"'+text.replace(/"/g,'""')+'"';
  }
  async function downloadData(manifestOnly=false){
    const rows=selection().slice();if(exporting||refreshing||!rows.length||[...enabled].some(id=>pending.has(id)||failed.has(id)||!loaded.has(id)))return;
    const format=$('download-format').value,metadata=downloadMetadata(rows);
    const basename=`typology-coruna-${grouped()?'objects':'observations'}-${new Date().toISOString().slice(0,10)}`;
    if(manifestOnly){saveBlob([JSON.stringify(metadata,null,2)],'application/json',basename+'-manifest.json');return;}
    exporting=true;$('download-data').disabled=true;notice(t('Preparing download…'));
    try{
      const parts=[],sample=downloadRecord(rows[0]),fields=Object.keys(sample);
      if(format==='csv')parts.push('\uFEFF'+fields.map(csvCell).join(',')+'\r\n');
      if(format==='geojson')parts.push('{"type":"FeatureCollection","metadata":'+JSON.stringify(metadata)+',"features":[');
      for(let offset=0;offset<rows.length;offset+=500){
        let chunk='';
        for(let i=offset;i<Math.min(rows.length,offset+500);i++){
          const record=downloadRecord(rows[i]);
          if(format==='csv')chunk+=fields.map(key=>csvCell(record[key])).join(',')+'\r\n';
          else if(format==='jsonl')chunk+=JSON.stringify(record)+'\n';
          else chunk+=(i?',':'')+JSON.stringify({type:'Feature',id:record.id,geometry:{type:'Point',coordinates:[record.longitude,record.latitude]},properties:record});
        }
        parts.push(chunk);await new Promise(resolve=>setTimeout(resolve,0));
      }
      if(format==='geojson')parts.push(']}');
      saveBlob(parts,{csv:'text/csv;charset=utf-8',geojson:'application/geo+json',jsonl:'application/x-ndjson'}[format],basename+'.'+format);notice(t('Download ready'));
    }catch(error){notice(t('Download failed. Try a smaller selection.'));}
    finally{exporting=false;refreshVisible();}
  }
  // URLs store browsing state, never data. Back restores filters and the selected object.
  function stateParams(){
    const f=currentFilters(),center=map.getCenter(),params=new URLSearchParams();
    params.set('lat',center.lat.toFixed(6));params.set('lon',center.lng.toFixed(6));params.set('z',map.getZoom());
    params.set('layers',f.categories.join(','));params.set('view',document.body.dataset.view);
    params.set('from',f.from);params.set('to',f.to);params.set('group',f.grouped?'1':'0');
    if(f.query)params.set('q',f.query);if(f.barrio)params.set('barrio',f.barrio);if(f.source)params.set('source',f.source);
    if(!f.unknown)params.set('unknown','0');if(f.area)params.set('area','1');if(f.cluster)params.set('cluster',f.cluster);
    if($('cluster-colors').checked)params.set('colors','1');
    if(selectedId){params.set('object',selectedId);if(curSign)params.set('observation',curSign.id);}
    return params;
  }
  function saveState(push=false){
    if(restoring||!ready||!map)return;
    const hash='#'+stateParams().toString();if(location.hash===hash)return;
    history[push?'pushState':'replaceState'](null,'',location.pathname+location.search+hash);
  }
  let restoreGeneration=0;
  async function restoreState(){
    const generation=++restoreGeneration;restoring=true;
    const p=new URLSearchParams(location.hash.slice(1));
    close(false);
    $('object-search').value=p.get('q')||'';
    $('barrio-filter').value=p.get('barrio')||'';$('source-filter').value=p.get('source')||'';
    $('area-filter').checked=p.get('area')==='1';$('group-duplicates').checked=p.get('group')!=='0';
    $('date-unknown').checked=p.get('unknown')!=='0';$('cluster-filter').value=p.get('cluster')||'';$('cluster-colors').checked=p.get('colors')==='1';
    const from=Number(p.get('from')),to=Number(p.get('to'));
    $('year-from').value=from>=1900?from:$('year-from').min;$('year-to').value=to>=1900?to:$('year-to').max;
    if(Number($('year-from').value)>Number($('year-to').value))$('year-to').value=$('year-from').value;yearLabels();
    enabled.clear();
    const defaults=manifest?['graffiti','street_name_plate']:['store_sign'];
    for(const id of (p.has('layers')?p.get('layers').split(','):defaults))if(categories.has(id))enabled.add(id);
    document.querySelectorAll('[data-category]').forEach(input=>input.checked=enabled.has(input.dataset.category));
    const lat=Number(p.get('lat')),lon=Number(p.get('lon')),zoom=Number(p.get('z'));
    if(p.has('lat')&&p.has('lon')&&Number.isFinite(lat)&&Number.isFinite(lon)&&Math.abs(lat)<=85&&Math.abs(lon)<=180)map.setView([lat,lon],zoom>=2&&zoom<=20?zoom:15);
    drawBarrio(!p.has('lat')&&!!$('barrio-filter').value);
    setView(p.get('view')==='wall'?'wall':'map',false);
    await Promise.all([...enabled].map(ensureLayer));if(generation!==restoreGeneration)return;
    updateSelection();
    const target=p.get('object');
    if(target){const row=active.find(row=>itemId(row)===target||row.id===target);if(row){browse=active;open(row,row.views?.find(v=>v.id===p.get('observation')),false);}}
    restoring=false;saveState();
  }

  const initial=await Promise.allSettled([fetchJSON('data.json'),fetchJSON('barrios.geojson'),fetchJSON('layers/manifest.json'),fetchJSON('crop-storage.json')]);
  if(initial[3].status==='fulfilled')cropStorage=CropStorage.create(initial[3].value);
  if(initial[0].status==='fulfilled')legacy=initial[0].value;
  if(initial[1].status==='fulfilled')prepareBarrios(initial[1].value);else $('barrio-status').textContent=t('Municipal barrio boundaries unavailable.');
  if(initial[2].status==='fulfilled')manifest=initial[2].value;
  const signs=legacy.signs.map(s=>decorate({...s,category:'store_sign',legacy_record:true,observation_id:s.source_crop_id,id:s.source_crop_id||s.id,score:s.score??s.s,crop:`img/${s.id}.jpg`,raw:s.ctx?`ctx/${s.id}.jpg`:null,source:s.src==='g'?'gsv':'apple',location_basis:'estimated_object',status:'legacy_detection'}));
  signs.fallback=true;loaded.set('store_sign',signs);
  for(const cluster of legacy.meta.clustering?.clusters||[])clusters.set(cluster.id,cluster);
  categories.set('store_sign',{id:'store_sign',label:'Shop signs',color:'#d79a3a',count:signs.length});
  for(const c of manifest?.categories||[])categories.set(c.id,c);
  for(const c of categories.values())addLayer(c);
  for(const cluster of clusters.values()){const option=document.createElement('option');option.value=cluster.id;option.textContent=`${cluster.label} · ${cluster.name}`;$('cluster-filter').append(option);}
  $('cluster-coverage').textContent=t('Cluster colors apply to previously clustered shop signs');
  for(const id of ['year-from','year-to']){$(id).min=2008;$(id).max=new Date().getFullYear();}
  $('year-from').value=2008;$('year-to').value=new Date().getFullYear();yearLabels();

  map=L.map('map',{zoomControl:true,minZoom:2,maxZoom:20}).setView([43.3665,-8.4065],15);
  const satellite=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:20,maxNativeZoom:19,attribution:'Imagery &copy; Esri, Maxar, Earthstar Geographics'}).addTo(map);
  const streets=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:20,maxNativeZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'});
  L.control.layers({[t('Satellite')]:satellite,[t('Streets')]:streets},null,{position:'topleft'}).addTo(map);
  L.control.scale({imperial:false,position:'bottomright'}).addTo(map);
  const ObservationCanvas=L.Layer.extend({
    onAdd(m){this.canvas=L.DomUtil.create('canvas','observation-canvas');m.getPane('overlayPane').append(this.canvas);this.groups=[];m.on('moveend zoomend resize',this.draw,this);m.on('click',this.click,this);this.draw();},
    onRemove(m){m.off('moveend zoomend resize',this.draw,this);m.off('click',this.click,this);this.canvas.remove();},
    draw(){
      const size=map.getSize(),dpr=window.devicePixelRatio||1,c=this.canvas;c.width=size.x*dpr;c.height=size.y*dpr;c.style.width=size.x+'px';c.style.height=size.y+'px';L.DomUtil.setPosition(c,map.containerPointToLayerPoint([0,0]));
      const ctx=c.getContext('2d');ctx.scale(dpr,dpr);const cells=new Map(),bounds=map.getBounds(),cell=map.getZoom()<17?18:10;
      for(const row of active){if(!bounds.contains([row.la,row.lo]))continue;const p=map.latLngToContainerPoint([row.la,row.lo]),key=`${Math.floor(p.x/cell)}:${Math.floor(p.y/cell)}`;let group=cells.get(key);if(!group){group={x:p.x,y:p.y,items:[]};cells.set(key,group);}group.items.push(row);}
      this.groups=[...cells.values()];
      for(const group of this.groups){
        const colors=[...new Set(group.items.map(color))],r=group.items.length>1?Math.min(12,5+Math.log2(group.items.length)):4;group.r=r;
        colors.forEach((col,i)=>{ctx.beginPath();ctx.moveTo(group.x,group.y);ctx.arc(group.x,group.y,r,i/colors.length*Math.PI*2,(i+1)/colors.length*Math.PI*2);ctx.closePath();ctx.fillStyle=col;ctx.fill();});
        ctx.strokeStyle=group.items.some(row=>itemId(row)===selectedId)?'#fff':'#141519';ctx.lineWidth=group.items.some(row=>itemId(row)===selectedId)?3:1;ctx.beginPath();ctx.arc(group.x,group.y,r,0,Math.PI*2);ctx.stroke();
        if(group.items.length>1){ctx.fillStyle='#fff';ctx.font='bold 10px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(group.items.length>99?'99+':String(group.items.length),group.x,group.y);}
      }
      $('map-status').hidden=false;$('map-status').textContent=`${this.groups.reduce((n,g)=>n+g.items.length,0).toLocaleString()} ${grouped()?t('items'):t('observations')} ${t('in view · click a dot to inspect')}`;
    },
    click(event){
      const p=event.containerPoint;let best=null,distance=Infinity;
      for(const group of this.groups){const d=Math.hypot(group.x-p.x,group.y-p.y);if(d<group.r+7&&d<distance){best=group;distance=d;}}
      if(!best)return;
      if(best.items.length===1){browse=active.filter(inMap);open(best.items[0]);return;}
      close();pointSelection=best.items;$('map-results').classList.remove('collapsed');$('results-toggle').textContent=t('Hide');$('results-toggle').setAttribute('aria-expanded','true');renderPanel();$('map-thumbnails').querySelector('button')?.focus();
    }
  });
  const observations=new ObservationCanvas().addTo(map);
  let mapTimer;
  map.on('moveend',()=>{pointSelection=null;refreshVisible();clearTimeout(mapTimer);mapTimer=setTimeout(()=>saveState(true),200);});
  new ResizeObserver(()=>map.invalidateSize()).observe($('map'));
  new IntersectionObserver(entries=>{if(entries[0].isIntersecting&&shown<order.length&&document.body.dataset.view==='wall')more();},{rootMargin:'600px'}).observe($('more'));
  $('results-more').onclick=morePanel;
  $('results-clear').onclick=()=>{pointSelection=null;renderPanel();};
  $('results-toggle').onclick=()=>{const collapsed=$('map-results').classList.toggle('collapsed');$('results-toggle').textContent=t(collapsed?'Show':'Hide');$('results-toggle').setAttribute('aria-expanded',String(!collapsed));if(!collapsed)renderPanel();};
  $('v-map').onclick=()=>setView('map');$('v-wall').onclick=()=>setView('wall');
  $('shuffle').onclick=()=>{setView('wall');for(let i=order.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[order[i],order[j]]=[order[j],order[i]];}renderWall();};
  $('full').onclick=()=>{full=!full;showPic();};$('close').onclick=()=>close();$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
  detail.onclick=event=>{if(event.target===detail&&document.body.dataset.view==='wall')close();};
  document.addEventListener('keydown',event=>{
    if(!detail.hasAttribute('open')||/INPUT|SELECT|TEXTAREA/.test(event.target.tagName))return;
    if(event.key==='Escape')close();
    if(event.key==='ArrowRight'){event.preventDefault();step(1);}if(event.key==='ArrowLeft'){event.preventDefault();step(-1);}if(event.key==='f'){full=!full;showPic();}
    if(event.key==='Tab'&&document.body.dataset.view==='wall'){
      const buttons=[...detail.querySelectorAll('button:not(:disabled),a[href]')].filter(el=>!el.hidden&&el.getClientRects().length);
      if(event.shiftKey&&document.activeElement===buttons[0]){event.preventDefault();buttons.at(-1)?.focus();}
      else if(!event.shiftKey&&document.activeElement===buttons.at(-1)){event.preventDefault();buttons[0]?.focus();}
    }
  });
  function changeFilters(){close(false);updateSelection();saveState(true);}
  for(const id of ['source-filter','area-filter','group-duplicates','date-unknown','cluster-filter'])$(id).onchange=changeFilters;
  $('barrio-filter').onchange=()=>{drawBarrio(true);changeFilters();};
  $('cluster-colors').onchange=()=>{renderWall();renderPanel();observations.draw();saveState();};
  for(const id of ['year-from','year-to'])$(id).oninput=()=>{if(Number($('year-from').value)>Number($('year-to').value))$(id==='year-from'?'year-to':'year-from').value=$(id).value;yearLabels();changeFilters();};
  let searchTimer;$('object-search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(changeFilters,200);};
  $('focus-results').onclick=()=>{if(!active.length)return;setView('map');map.fitBounds(L.latLngBounds(active.map(row=>[row.la,row.lo])),{padding:[40,40],maxZoom:18});};
  $('reset-filters').onclick=()=>{
    $('object-search').value='';$('barrio-filter').value='';$('source-filter').value='';$('area-filter').checked=false;$('date-unknown').checked=true;
    $('year-from').value=$('year-from').min;$('year-to').value=$('year-to').max;$('cluster-filter').value='';yearLabels();drawBarrio();changeFilters();
  };
  $('layers-all').onclick=async()=>{for(const id of categories.keys())enabled.add(id);document.querySelectorAll('[data-category]').forEach(input=>input.checked=true);await Promise.all([...enabled].map(ensureLayer));updateSelection();};
  $('layers-none').onclick=()=>{enabled.clear();document.querySelectorAll('[data-category]').forEach(input=>input.checked=false);changeFilters();};
  $('download-data').onclick=()=>downloadData();$('download-manifest').onclick=()=>downloadData(true);
  $('share').onclick=async()=>{saveState();try{await navigator.clipboard.writeText(location.href);notice(t('View link copied'));}catch(error){window.prompt(t('Copy this view link'),location.href);}};
  window.addEventListener('popstate',restoreState);
  $('snapshot-update').onclick=async()=>{
    if(!availableManifest||detail.hasAttribute('open')||pending.size)return;
    refreshing=true;
    const controls=[...document.querySelectorAll('.object-layers input,.object-layers select,.object-layers button')];
    const disabled=controls.map(control=>control.disabled);controls.forEach(control=>control.disabled=true);
    refreshVisible();
    const button=$('snapshot-update');button.disabled=true;$('update-status').textContent=t('Updating selected layers…');
    const next=availableManifest;
    try{
      const staged=await Promise.all(next.categories.filter(c=>enabled.has(c.id)).map(async c=>[c.id,parseLayer(c,await fetchJSON(c.url))]));
      manifest=next;
      for(const c of next.categories){if(!categories.has(c.id))addLayer(c);categories.set(c.id,c);loaded.delete(c.id);}
      if(!next.categories.some(c=>c.id==='store_sign'))loaded.set('store_sign',signs);
      for(const [id,rows]of staged){loaded.set(id,rows);failed.delete(id);}
      if(availableManifest?.snapshot_id===next.snapshot_id)availableManifest=null;
      button.hidden=!availableManifest;updateYears();updateSelection();$('update-status').textContent='';
    }catch(error){$('update-status').textContent=t('Update failed. Your current data is unchanged; try again.');}
    finally{refreshing=false;controls.forEach((control,index)=>control.disabled=disabled[index]);button.disabled=false;refreshVisible();}
  };
  setInterval(async()=>{try{const next=await fetchJSON('layers/manifest.json');if(manifest&&next.snapshot_id!==manifest.snapshot_id){availableManifest=next;$('snapshot-update').hidden=false;}}catch(error){}},60000);
  if(!manifest)$('layer-error').textContent=t('Object layers are unavailable. Saved shop signs are still available.');
  ready=true;await restoreState();
})();
