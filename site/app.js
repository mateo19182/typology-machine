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
    'Bounds: Concello da Coruña':'Límites: Concello da Coruña','Image previews on map':'Vistas previas en el mapa','Loading map…':'Cargando mapa…','Preview':'Vista previa','Clear selection':'Quitar selección',
    'Record details':'Detalles del registro','records, with matching viewpoints':'registros, con sus vistas coincidentes',
    'Reload the page to retry.':'Recarga la página para reintentar.',
    'Some selected layers failed to load. Retry before downloading.':'Algunas capas no se cargaron. Reinténtalo antes de descargar.'
  });
  Object.assign(translations, {'Experiments':'Experimentos','Minimum confidence':'Confianza mínima','Show on map':'Ver en mapa','Filters':'Filtros','Categories':'Categorías','Date and source':'Fecha y fuente','Display options':'Opciones de visualización','From':'Desde','To':'Hasta','Close filters':'Cerrar filtros','Shuffle wall':'Mezclar galería','Copy link':'Copiar enlace','Capture date known':'Fecha de captura conocida','All dates':'Todas las fechas','No categories':'Sin categorías'});
  const t=value=>language==='es'?(translations[value]||value):value;
  document.documentElement.lang=language;
  // Translate interface nodes only; detected text and provider metadata stay original.
  const $ = id => document.getElementById(id), wall = $('wall');
  for(const element of document.querySelectorAll('nav, .browse-tools, .detail .arrows, .map-results')){
    const walker=document.createTreeWalker(element,NodeFilter.SHOW_TEXT);let node;
    while(node=walker.nextNode()){const raw=node.nodeValue,trimmed=raw.trim();if(translations[trimmed])node.nodeValue=raw.replace(trimmed,t(trimmed));}
  }
  for(const element of document.querySelectorAll('[aria-label],[placeholder]'))for(const attr of ['aria-label','placeholder'])if(element.hasAttribute(attr))element.setAttribute(attr,t(element.getAttribute(attr)));
  $('d-map').textContent=t('map');$('map-loading-text').textContent=t('Loading map…');
  $('language').textContent=language==='es'?'EN':'ES';$('language').onclick=()=>{try{localStorage.setItem('typology-language',language==='es'?'en':'es');}catch(error){}location.reload();};
  if (matchMedia('(max-width:600px)').matches) $('object-layers').open=false;
  let map, order=[], browse=[], curSign, curObject, full=false;
  let manifest=null, availableManifest=null, selectedId='', restoring=true, ready=false, refreshing=false;
  let panelItems=[], pointSelection=null, barrioLayer=null;
  let cellData=null, pendingCells=null, cellController, cellGeneration=0, result=null, hasMore=false, pageBusy=false, queryBusy=false, queryError=false, shuffleSeed=0;
  let queryController, queryGeneration=0, detailGeneration=0, viewport;
  let barrios=[];
  let cropStorage=CropStorage.create();
  const enabled=new Set();
  const categories=new Map(), clusters=new Map();
  const grouped=()=>$('group-duplicates').checked;
  const label=row=>t(categories.get(row.category)?.label||row.category);
  const color=row=>$('cluster-colors').checked&&row.legacy_record?row.cluster_color||'#777':categories.get(row.category)?.color||'#ddd';
  const sourceOf=row=>row.source||(row.src==='g'?'gsv':'apple');
  const itemId=row=>row.object_id||row.id;
  const detail=$('detail');
  let returnFocus=null, statusTimer;
  function notice(message){$('action-status').textContent=message;clearTimeout(statusTimer);statusTimer=setTimeout(()=>$('action-status').textContent='',5000);}
  // Small cached JPEGs from the API; the original crop is the fallback.
  function thumbImage(img,path,size){
    img.onload=()=>img.classList.add('loaded');
    if(typeof path!=='string'||!/^\/(img|proposal-assets)\/[A-Za-z0-9_./-]+\.jpg$/.test(path)||path.split('/').includes('..')){cropStorage.image(img,path);return;}
    img.onerror=()=>cropStorage.image(img,path);img.src=`/api/thumb/${size}${path}`;
  }
  async function fetchJSON(url,options){const response=await fetch(url,options);if(!response.ok)throw Error(`${url}: ${response.status}`);return response.json();}

  // Boundaries are only used to draw the selected barrio. Membership is indexed on the server.
  function prepareBarrios(data){
    barrios=data.features.map(feature=>({id:String(feature.properties.id),name:feature.properties.name,feature})).sort((a,b)=>a.name.localeCompare(b.name,'gl'));
    for(const barrio of barrios){const option=document.createElement('option');option.value=barrio.id;option.textContent=barrio.name;$('barrio-filter').append(option);}
    const outside=document.createElement('option');outside.value='outside';outside.textContent=t('Outside mapped barrios');$('barrio-filter').append(outside);
    $('barrio-filter').disabled=false;
    const attribution=document.createElement('a');attribution.href=data.metadata.source;attribution.target='_blank';attribution.rel='noopener';attribution.textContent=t('Bounds: Concello da Coruña');$('barrio-status').replaceChildren(attribution);
  }
  function currentFilters(){return {
    categories:[...enabled].sort(),query:$('object-search').value.trim(),barrio:$('barrio-filter').value,
    confidence:$('confidence-filter').value,source:$('source-filter').value,from:$('year-from').value,to:$('year-to').value,
    unknown:$('date-unknown').checked,grouped:grouped(),area:$('area-filter').checked,
    cluster:$('cluster-filter').value
  };}
  function yearLabels(){$('year-from-value').textContent=$('year-from').value;$('year-to-value').textContent=$('year-to').value;}
  function mapViewport(){
    const b=map.getBounds(),size=map.getSize(),cell=map.getZoom()<17?18:10;
    if(size.x&&size.y)viewport={bbox:[Math.max(-180,b.getWest()),Math.max(-85,b.getSouth()),Math.min(180,b.getEast()),Math.min(85,b.getNorth())].join(','),cols:Math.ceil(size.x/cell),rows:Math.ceil(size.y/cell)};
    return viewport;
  }
  function queryParams(){
    const f=currentFilters(),view=mapViewport();
    const params=new URLSearchParams({snapshot:manifest.query_snapshot,layers:f.categories.join(','),q:f.query,
      confidence:f.confidence,barrio:f.barrio,source:f.source,from:f.from,to:f.to,unknown:f.unknown?'1':'0',group:f.grouped?'1':'0',
      area:f.area?'1':'0',cluster:f.cluster,colors:$('cluster-colors').checked?'1':'0',
      view:document.body.dataset.view,map:'0',limit:48,seed:shuffleSeed});
    params.set('bbox',view.bbox);
    if(pointSelection)params.set('list_bbox',pointSelection.bounds.join(','));
    return params;
  }
  function updateSelection(snapshot){pointSelection=null;return refreshVisible(snapshot);}
  function updateCounts(){
    const busy=queryBusy||refreshing,total=result?.total||0;
    // Loading text must not wrap the mobile masthead: resizing the map emits moveend and starts another query.
    $('count').textContent=`${total.toLocaleString()} ${grouped()?t('items'):t('observations')}`;
    $('count').setAttribute('aria-busy',String(busy));
    $('count').title=busy?t('Loading selected layers…'):'';
    $('layer-summary').textContent=`${t('Filters')} · ${enabled.size}`;
    const f=currentFilters(), names=[...enabled].map(id=>t(categories.get(id)?.label||id));
    const summary=[`${t('Minimum confidence')} ${f.confidence}`,names.join(', ')||t('No categories'),f.query?`“${f.query}”`:null,barrios.find(b=>b.id===f.barrio)?.name||(f.barrio==='outside'?t('Outside mapped barrios'):null),f.source?$('source-filter').selectedOptions[0].textContent:null,f.from!==$('year-from').min||f.to!==$('year-to').max?`${f.from}–${f.to}`:null,!f.unknown?t('Capture date known'):null,f.area?t('Only this map area'):null,!f.grouped?t('observations'):null].filter(Boolean).join(' · ');
    $('filter-summary').textContent=summary;$('filter-summary').title=summary;
    $('cluster-controls').hidden=!enabled.has('store_sign')||!clusters.size;
    $('focus-results').disabled=busy||!total||queryError;
    $('download-data').disabled=$('download-manifest').disabled=busy||queryError||!total;
    $('download-summary').textContent=queryError?t('Some selected layers failed to load. Retry before downloading.'):busy?t('Loading selected layers…'):`${total.toLocaleString()} ${grouped()?t('Selected objects'):t('Selected observations')}`;
    for(const c of categories.values()){
      const count=document.querySelector(`[data-category-count="${c.id}"]`);
      if(count){count.textContent=(enabled.has(c.id)?result?.counts[c.id]||0:c.count||0).toLocaleString();count.title=t(enabled.has(c.id)?'Selected objects':'Total in snapshot; enable layer to filter by capture date');}
    }
  }
  // Map cells sit on a world pixel grid and cover more than the viewport: panning redraws
  // cached cells at once, and only leaving the loaded area asks the server again.
  function cellGrid(pad){
    const z=map.getZoom(),size=z<17?18:10,view=map.getPixelBounds(),world=map.getPixelWorldBounds(z);
    const range=(lo,hi,min,max)=>{const a=Math.floor(lo/size),b=Math.ceil(hi/size),extra=Math.max(0,Math.min(Math.ceil((b-a)*pad),Math.floor((800-(b-a))/2)));return [Math.max(Math.ceil(min/size),a-extra),Math.min(Math.floor(max/size),b+extra)];};
    const [gx0,gx1]=range(view.min.x,view.max.x,world.min.x,world.max.x),[gy0,gy1]=range(view.min.y,view.max.y,map.project([85,0],z).y,map.project([-85,0],z).y);
    return {z,size,gx0,gx1,gy0,gy1};
  }
  const covers=(a,b)=>a.z===b.z&&a.size===b.size&&a.gx0<=b.gx0&&a.gx1>=b.gx1&&a.gy0<=b.gy0&&a.gy1>=b.gy1;
  function cellParams(snapshot){const params=queryParams();for(const key of ['bbox','cols','rows','view','map','limit','seed','list_bbox','area'])params.delete(key);params.set('snapshot',snapshot.query_snapshot);return params;}
  async function refreshCells(snapshot=manifest){
    if(!map||!snapshot||document.body.dataset.view!=='map')return;
    const key=cellParams(snapshot).toString(),view=cellGrid(0);
    if(view.gx1<=view.gx0||view.gy1<=view.gy0)return;
    for(const loaded of [cellData,pendingCells])if(loaded?.key===key&&covers(loaded,view))return;
    const grid={...cellGrid(.5),key},generation=++cellGeneration,edge=(x,y)=>map.unproject([x*grid.size,y*grid.size],grid.z);
    cellController?.abort();cellController=new AbortController();pendingCells=grid;$('map-loading').hidden=false;
    const nw=edge(grid.gx0,grid.gy0),se=edge(grid.gx1,grid.gy1),params=cellParams(snapshot);
    params.set('bbox',[Math.max(-180,nw.lng),Math.max(-85,se.lat),Math.min(180,se.lng),Math.min(85,nw.lat)].join(','));
    params.set('cols',grid.gx1-grid.gx0);params.set('rows',grid.gy1-grid.gy0);
    try{
      const data=await fetchJSON('/api/cells?'+params,{signal:cellController.signal});
      if(generation!==cellGeneration)return;
      // The server counts rows from the south edge; the screen grid counts from the north.
      grid.cells=data.cells.map(([cx,cy,count,lo,la,colors,id,sample,crop])=>{
        const gx=grid.gx0+cx,gy=grid.gy1-1-cy,a=edge(gx,gy),b=edge(gx+1,gy+1);
        return {count,lo,la,colors,id,sample,crop,bounds:[a.lng,b.lat,b.lng,a.lat]};
      });
      cellData=grid;observations.draw();
    }catch(error){if(generation===cellGeneration&&error.name!=='AbortError'){$('map-status').hidden=false;$('map-status').textContent=t('Could not load');}}
    finally{if(generation===cellGeneration){pendingCells=null;$('map-loading').hidden=true;}}
  }
  let merged=null;
  function displayCells(){
    if(!cellData)return [];
    const z=map.getZoom();if(z>=cellData.z)return cellData.cells;
    if(merged?.data===cellData&&merged.z===z)return merged.cells;
    // Zooming out merges cached cells immediately; the exact grid replaces them when it arrives.
    const size=z<17?18:10,groups=new Map();
    for(const c of cellData.cells){
      const p=map.project([c.la,c.lo],z),key=Math.floor(p.x/size)+','+Math.floor(p.y/size),g=groups.get(key);
      if(!g){groups.set(key,{...c,colors:[...c.colors],lo:c.lo*c.count,la:c.la*c.count,top:c.count});continue;}
      g.lo+=c.lo*c.count;g.la+=c.la*c.count;g.count+=c.count;g.id=null;
      for(const color of c.colors)if(!g.colors.includes(color))g.colors.push(color);
      g.bounds=[Math.min(g.bounds[0],c.bounds[0]),Math.min(g.bounds[1],c.bounds[1]),Math.max(g.bounds[2],c.bounds[2]),Math.max(g.bounds[3],c.bounds[3])];
      if(c.crop&&(c.count>g.top||!g.crop)){g.sample=c.sample;g.crop=c.crop;g.top=c.count;}
    }
    for(const g of groups.values()){g.lo/=g.count;g.la/=g.count;}
    merged={data:cellData,z,cells:[...groups.values()]};return merged.cells;
  }
  function updateVisible(){
    if(!map)return;
    let visible=0;const bounds=map.getBounds();
    for(const cell of cellData?.cells||[])if(bounds.contains([cell.la,cell.lo]))visible+=cell.count;
    if(document.body.dataset.view==='map'){$('map-status').hidden=false;$('map-status').textContent=`${visible.toLocaleString()} ${grouped()?t('items'):t('observations')} ${t('in view · click a dot to inspect')}`;}
    $('results-title').textContent=`${t(pointSelection?'At this point':'In this area')} · ${(pointSelection?.count??visible).toLocaleString()}`;
  }
  async function refreshVisible(snapshot=manifest){
    if(!map||!manifest)return false;
    const generation=++queryGeneration;
    refreshCells(snapshot);
    queryController?.abort();queryController=new AbortController();queryBusy=true;pageBusy=false;updateCounts();
    try{
      const params=queryParams();params.set('snapshot',snapshot.query_snapshot);
      const next=await fetchJSON('/api/query?'+params,{signal:queryController.signal});
      if(generation!==queryGeneration)return false;
      manifest=snapshot;result=next;hasMore=next.has_more;queryError=false;
      order=document.body.dataset.view==='wall'?next.items:[];
      panelItems=document.body.dataset.view==='map'?next.items:[];
      $('layer-error').textContent='';renderWall();renderPanel();return true;
    }catch(error){
      if(generation!==queryGeneration||error.name==='AbortError')return false;
      queryError=true;$('layer-error').textContent=t('Could not load')+'. '+t('Select it to retry.');return false;
    }finally{if(generation===queryGeneration){queryBusy=false;updateCounts();saveState();}}
  }
  async function loadMore(){
    if(!hasMore||pageBusy||queryBusy||queryError)return;
    const generation=queryGeneration,isWall=document.body.dataset.view==='wall',items=isWall?order:panelItems;
    const params=queryParams();params.set('offset',items.length);params.set('map','0');
    pageBusy=true;$('results-more').disabled=true;
    try{
      const next=await fetchJSON('/api/query?'+params,{signal:queryController.signal});
      if(generation!==queryGeneration)return;
      items.push(...next.items);hasMore=next.has_more;
      const target=isWall?wall:$('map-thumbnails');
      for(const row of next.items)target.append(tile(row,items));
      $('results-more').hidden=!hasMore;
      if(isWall&&hasMore&&$('more').getBoundingClientRect().top<innerHeight+600)setTimeout(loadMore,0);
    }catch(error){if(error.name!=='AbortError')notice(t('Could not load')+'. '+t('Select it to retry.'));}
    finally{if(generation===queryGeneration){pageBusy=false;$('results-more').disabled=false;}}
  }
  function addLayer(c){
    const row=document.createElement('label');row.className='layer-option';
    const input=document.createElement('input');input.type='checkbox';input.dataset.category=c.id;input.checked=enabled.has(c.id);
    const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=c.color;
    const name=document.createElement('span');name.textContent=t(c.label);
    const count=document.createElement('small');count.dataset.categoryCount=c.id;count.textContent=(c.count||0).toLocaleString();
    row.append(input,swatch,name,count);$('layer-options').append(row);
    input.onchange=()=>{if(input.checked)enabled.add(c.id);else enabled.delete(c.id);changeFilters();};
  }
  function drawBarrio(fit=false){
    if(barrioLayer){map.removeLayer(barrioLayer);barrioLayer=null;}
    const barrio=barrios.find(b=>b.id===$('barrio-filter').value);
    if(barrio){barrioLayer=L.geoJSON(barrio.feature,{style:{color:'#e8bc6a',weight:2,fillOpacity:.04},interactive:false}).addTo(map);if(fit)map.fitBounds(barrioLayer.getBounds(),{padding:[35,35],maxZoom:17});}
  }
  function tile(row,items){
    const button=document.createElement('button');button.className='tile';button.dataset.objectId=itemId(row);button.style.borderBottomColor=color(row);
    button.setAttribute('aria-label',[label(row),row.text||row.t,row.barrio].filter(Boolean).join(' · '));button.setAttribute('aria-pressed',String(itemId(row)===selectedId));
    const image=document.createElement('img');image.loading='lazy';thumbImage(image,row.crop,384);image.alt='';button.append(image);
    if(row.view_count>1){const badge=document.createElement('span');badge.className='view-count';badge.textContent=`${row.view_count} ${t('views')}`;button.append(badge);}
    button.onclick=()=>{browse=items;open(row);};return button;
  }
  function renderWall(){
    if(document.body.dataset.view!=='wall')return;
    wall.replaceChildren();for(const row of order)wall.append(tile(row,order));
    if(!order.length){const empty=document.createElement('p');empty.textContent=t('No objects match these filters.');wall.append(empty);}
    if(hasMore&&$('more').getBoundingClientRect().top<innerHeight+600)setTimeout(loadMore,0);
  }
  function renderPanel(){
    if(document.body.dataset.view!=='map')return;
    $('map-thumbnails').replaceChildren();
    updateVisible();
    $('results-clear').hidden=!pointSelection;$('results-empty').textContent=panelItems.length?'':t('No objects in this area. Move the map or change filters.');
    if(!$('map-results').classList.contains('collapsed'))for(const row of panelItems)$('map-thumbnails').append(tile(row,panelItems));
    $('results-more').hidden=!hasMore;$('results-more').disabled=pageBusy;
  }
  function setView(view,save=true){
    mapViewport();const changed=document.body.dataset.view!==view;
    document.body.dataset.view=view;$('v-map').setAttribute('aria-pressed',String(view==='map'));$('v-wall').setAttribute('aria-pressed',String(view==='wall'));
    detail.setAttribute('aria-modal',String(view==='wall'));
    if(view==='map')map.invalidateSize();
    if(changed&&save)updateSelection();
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
  async function open(object,view,save=true){
    const generation=++detailGeneration;
    if(!object.views){
      try{
        const params=queryParams();params.set('id',itemId(object));
        object=await fetchJSON('/api/object?'+params);
        if(generation!==detailGeneration)return;
      }catch(error){if(generation===detailGeneration)notice(t('Could not load')+'. '+t('Select it to retry.'));return;}
    }
    if(typeof view==='string')view=object.views.find(v=>v.id===view);

    if(!detail.hasAttribute('open'))returnFocus=document.activeElement;
    curObject=object;curSign=view||object;selectedId=itemId(object);const row=curSign;
    const views=object.views||[object],index=browse.findIndex(item=>itemId(item)===itemId(object)),viewIndex=views.findIndex(v=>v.id===row.id);
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
  function close(save=true){detailGeneration++;detail.removeAttribute('open');selectedId='';curObject=null;curSign=null;observations.draw();if(returnFocus?.isConnected)returnFocus.focus();if(save)saveState();}
  function step(delta){if(!browse.length||!curObject)return;const index=browse.findIndex(item=>itemId(item)===itemId(curObject));open(browse[(Math.max(0,index)+delta+browse.length)%browse.length]);}

  function downloadData(manifestOnly=false){
    if(queryBusy||refreshing||queryError||!result?.total)return;
    const params=queryParams();params.set('format',manifestOnly?'manifest':$('download-format').value);
    // The server streams the entire selection. Gallery pagination never limits exports.
    const anchor=document.createElement('a');anchor.href='/api/export?'+params;anchor.download='';document.body.append(anchor);anchor.click();anchor.remove();
    notice(t('Preparing download…'));
  }
  // URLs store browsing state, never data. Back restores filters and the selected object.
  function stateParams(){
    const f=currentFilters(),center=map.getCenter(),params=new URLSearchParams();
    params.set('lat',center.lat.toFixed(6));params.set('lon',center.lng.toFixed(6));params.set('z',map.getZoom());
    params.set('layers',f.categories.join(','));params.set('view',document.body.dataset.view);
    params.set('confidence',f.confidence);params.set('from',f.from);params.set('to',f.to);params.set('group',f.grouped?'1':'0');
    if(shuffleSeed)params.set('seed',shuffleSeed);
    if(f.query)params.set('q',f.query);if(f.barrio)params.set('barrio',f.barrio);if(f.source)params.set('source',f.source);
    if(!f.unknown)params.set('unknown','0');if(f.area)params.set('area','1');if(f.cluster)params.set('cluster',f.cluster);
    if($('cluster-colors').checked)params.set('colors','1');if(!$('map-previews').checked)params.set('previews','0');
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
    shuffleSeed=Math.max(0,Math.min(2147483646,parseInt(p.get('seed'),10)||0));
    $('confidence-filter').value=['0.5','0.65','0.75','0.85'].includes(p.get('confidence'))?p.get('confidence'):'0.65';
    $('object-search').value=p.get('q')||'';
    $('barrio-filter').value=p.get('barrio')||'';$('source-filter').value=p.get('source')||'';
    $('area-filter').checked=p.get('area')==='1';$('group-duplicates').checked=p.get('group')!=='0';
    $('date-unknown').checked=p.get('unknown')!=='0';$('cluster-filter').value=p.get('cluster')||'';$('cluster-colors').checked=p.get('colors')==='1';$('map-previews').checked=p.get('previews')!=='0';
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
    await updateSelection();if(generation!==restoreGeneration)return;
    const target=p.get('object');
    if(target){browse=document.body.dataset.view==='wall'?order:panelItems;await open({id:target},p.get('observation'),false);}
    restoring=false;saveState();
  }

  const initial=await Promise.allSettled([fetchJSON('/api/meta'),fetchJSON('barrios.geojson'),fetchJSON('crop-storage.json')]);
  if(initial[2].status==='fulfilled')cropStorage=CropStorage.create(initial[2].value);
  if(initial[1].status==='fulfilled')prepareBarrios(initial[1].value);else $('barrio-status').textContent=t('Municipal barrio boundaries unavailable.');
  if(initial[0].status!=='fulfilled'){$('layer-error').textContent=t('Could not load')+'. '+t('Reload the page to retry.');return;}
  manifest=initial[0].value;
  for(const cluster of manifest.legacy.clustering?.clusters||[])clusters.set(cluster.id,cluster);
  for(const c of manifest.categories)categories.set(c.id,c);
  for(const c of categories.values())addLayer(c);
  for(const cluster of clusters.values()){const option=document.createElement('option');option.value=cluster.id;option.textContent=`${cluster.label} · ${cluster.name}`;$('cluster-filter').append(option);}
  $('cluster-coverage').textContent=t('Cluster colors apply to previously clustered shop signs');
  for(const id of ['year-from','year-to']){$(id).min=Math.min(2008,manifest.min_year);$(id).max=Math.max(new Date().getFullYear(),manifest.max_year);}
  $('year-from').value=$('year-from').min;$('year-to').value=$('year-to').max;yearLabels();

  map=L.map('map',{zoomControl:true,minZoom:2,maxZoom:20}).setView([43.3665,-8.4065],15);
  const satellite=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:20,maxNativeZoom:19,attribution:'Imagery &copy; Esri, Maxar, Earthstar Geographics'}).addTo(map);
  const streets=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:20,maxNativeZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'});
  L.control.layers({[t('Satellite')]:satellite,[t('Streets')]:streets},null,{position:'topleft'}).addTo(map);
  L.control.scale({imperial:false,position:'bottomright'}).addTo(map);
  const ObservationLayer=L.Layer.extend({
    onAdd(m){
      this.root=L.DomUtil.create('div','observation-layer leaflet-zoom-animated',m.getPane('overlayPane'));
      this.canvas=L.DomUtil.create('canvas','observation-canvas',this.root);
      this.focus=L.DomUtil.create('div','map-focus',this.root);this.focus.hidden=true;
      this.focus.innerHTML='<span class="pulse"></span><span class="card"><img alt=""></span>';this.focusImg=this.focus.querySelector('img');
      this.previews=new Map();this.groups=[];this.frame=0;
      m.on('moveend zoomend resize viewreset',this.draw,this);m.on('move',this.schedule,this);m.on('zoomanim',this.animate,this);m.on('click',this.click,this);
      this.draw();
    },
    onRemove(m){m.off('moveend zoomend resize viewreset',this.draw,this);m.off('move',this.schedule,this);m.off('zoomanim',this.animate,this);m.off('click',this.click,this);cancelAnimationFrame(this.frame);this.root.remove();},
    // Dragging redraws cached cells every frame, so dots never wait for the server to follow the map.
    schedule(){if(!this.frame)this.frame=requestAnimationFrame(()=>{this.frame=0;if(map._animatingZoom)return;this.draw();if(Number.isInteger(map.getZoom()))refreshCells();});},
    // Same transform as Leaflet's own renderers: the drawing scales with the zoom animation.
    animate(e){const scale=map.getZoomScale(e.zoom,this.zoom),offset=map.project(this.center,e.zoom).subtract(map.getSize().multiplyBy(scale/2)).subtract(map._getNewPixelOrigin(e.center,e.zoom));L.DomUtil.setTransform(this.root,offset,scale);},
    draw(){
      if(!this.root)return;
      const size=map.getSize(),dpr=window.devicePixelRatio||1,c=this.canvas;
      this.center=map.getCenter();this.zoom=map.getZoom();L.DomUtil.setPosition(this.root,map.containerPointToLayerPoint([0,0]));
      if(c.width!==Math.round(size.x*dpr)||c.height!==Math.round(size.y*dpr)){c.width=Math.round(size.x*dpr);c.height=Math.round(size.y*dpr);c.style.width=size.x+'px';c.style.height=size.y+'px';}
      const ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,size.x,size.y);
      const focus=document.body.dataset.view==='map'&&Number.isFinite(curSign?.la)&&Number.isFinite(curSign?.lo)?curSign:null;
      this.groups=[];
      for(const cell of displayCells()){
        const p=map.latLngToContainerPoint([cell.la,cell.lo]);
        if(p.x>-20&&p.y>-20&&p.x<size.x+20&&p.y<size.y+20)this.groups.push({cell,x:p.x,y:p.y,r:cell.count>1?Math.min(12,5+Math.log2(cell.count)):4});
      }
      ctx.font='bold 10px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
      for(const {cell,x,y,r} of this.groups){
        const selected=cell.id&&cell.id===selectedId;ctx.globalAlpha=focus&&!selected?.3:1;
        cell.colors.forEach((col,i)=>{ctx.beginPath();ctx.moveTo(x,y);ctx.arc(x,y,r,i/cell.colors.length*Math.PI*2,(i+1)/cell.colors.length*Math.PI*2);ctx.closePath();ctx.fillStyle=col;ctx.fill();});
        ctx.strokeStyle=selected?'#fff':'#141519';ctx.lineWidth=selected?3:1;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.stroke();
        if(cell.count>1){ctx.fillStyle='#fff';ctx.fillText(cell.count>99?'99+':String(cell.count),x,y);}
      }
      ctx.globalAlpha=1;
      if(focus){
        // Other viewpoints of the same object are hollow rings; the open one gets the halo.
        ctx.strokeStyle='#d79a3a';ctx.lineWidth=2;
        for(const view of curObject?.views||[])if(view!==focus&&Number.isFinite(view.la)){const p=map.latLngToContainerPoint([view.la,view.lo]);ctx.beginPath();ctx.arc(p.x,p.y,6,0,Math.PI*2);ctx.stroke();}
        const p=map.latLngToContainerPoint([focus.la,focus.lo]);
        ctx.fillStyle='rgba(10,11,13,.55)';ctx.beginPath();ctx.arc(p.x,p.y,18,0,Math.PI*2);ctx.fill();
        ctx.fillStyle='#d79a3a';ctx.beginPath();ctx.arc(p.x,p.y,9,0,Math.PI*2);ctx.fill();
        ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke();ctx.strokeStyle='#000';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(p.x,p.y,11,0,Math.PI*2);ctx.stroke();
      }
      this.place(size,focus);updateVisible();
    },
    place(size,focus){
      const chosen=[],at=focus&&map.latLngToContainerPoint([focus.la,focus.lo]);
      if($('map-previews').checked&&document.body.dataset.view==='map'){
        // Biggest clusters first; each preview keeps clear of the others and of the selection card.
        const candidates=this.groups.filter(g=>g.cell.crop&&g.y>64&&g.x>24&&g.x<size.x-24).sort((a,b)=>b.cell.count-a.cell.count||(a.cell.sample<b.cell.sample?-1:1));
        const limit=size.x<600?12:36;
        for(const g of candidates){
          if(chosen.length>=limit)break;
          if(at&&Math.abs(at.x-g.x)<90&&g.y-at.y>-80&&g.y-at.y<130)continue;
          if(chosen.every(o=>Math.abs(o.x-g.x)>=56||Math.abs(o.y-g.y)>=62))chosen.push(g);
        }
      }
      const keep=new Set();
      for(const g of chosen){
        const key=g.cell.sample;keep.add(key);let el=this.previews.get(key);
        if(!el){
          el=L.DomUtil.create('button','map-preview',this.root);el.type='button';const img=document.createElement('img');img.alt='';thumbImage(img,g.cell.crop,128);el.append(img);
          L.DomEvent.on(el,'click',event=>{L.DomEvent.stop(event);if(map.dragging?.moved())return;browse=panelItems;open({id:el.cell.sample},el.cell.sample);});
          L.DomEvent.on(el,'dblclick',L.DomEvent.stop);
          this.previews.set(key,el);
        }
        el.cell=g.cell;el.style.left=g.x+'px';el.style.top=(g.y-g.r-7)+'px';el.style.setProperty('--edge',g.cell.colors[0]);
        el.setAttribute('aria-label',`${t('Preview')} · ${g.cell.count.toLocaleString()} ${grouped()?t('items'):t('observations')}`);
      }
      for(const [key,el] of this.previews)if(!keep.has(key)){el.remove();this.previews.delete(key);}
      this.root.classList.toggle('has-focus',!!focus);this.focus.hidden=!focus;
      if(focus){
        this.focus.style.left=at.x+'px';this.focus.style.top=at.y+'px';
        if(this.focus.dataset.crop!==(focus.crop||'')){this.focus.dataset.crop=focus.crop||'';this.focusImg.hidden=!focus.crop;if(focus.crop){this.focusImg.onload=()=>this.draw();cropStorage.image(this.focusImg,focus.crop);}}
        // Keep the card inside the map; the stem still points at the observation.
        const card=this.focus.lastElementChild,w=card.offsetWidth,h=card.offsetHeight,left=at.x-w/2;
        card.style.setProperty('--shift',Math.max(8-left,Math.min(0,size.x-8-left-w))+'px');this.focus.classList.toggle('below',at.y<h+40);
      }
    },
    click(event){
      const p=event.containerPoint;let best=null,distance=Infinity;
      for(const g of this.groups){const d=Math.hypot(g.x-p.x,g.y-p.y);if(d<g.r+7&&d<distance){best=g;distance=d;}}
      if(!best)return;const cell=best.cell;
      if(cell.count===1&&cell.id){browse=panelItems;open({id:cell.id});return;}
      close();pointSelection={bounds:cell.bounds,count:cell.count};$('map-results').classList.remove('collapsed');$('results-toggle').textContent=t('Hide');$('results-toggle').setAttribute('aria-expanded','true');refreshVisible().then(()=>$('map-thumbnails').querySelector('button')?.focus());
    }
  });
  const observations=new ObservationLayer().addTo(map);
  let mapTimer;
  map.on('moveend',()=>{if(restoring||document.body.dataset.view!=='map')return;pointSelection=null;refreshCells();clearTimeout(mapTimer);mapTimer=setTimeout(()=>{refreshVisible();saveState(true);},200);});
  new ResizeObserver(()=>{if(document.body.dataset.view==='map')map.invalidateSize();}).observe($('map'));
  new IntersectionObserver(entries=>{if(entries[0].isIntersecting&&hasMore&&document.body.dataset.view==='wall')loadMore();},{rootMargin:'600px'}).observe($('more'));
  $('results-more').onclick=loadMore;
  $('results-clear').onclick=()=>{pointSelection=null;refreshVisible();};
  $('results-toggle').onclick=()=>{const collapsed=$('map-results').classList.toggle('collapsed');$('results-toggle').textContent=t(collapsed?'Show':'Hide');$('results-toggle').setAttribute('aria-expanded',String(!collapsed));if(!collapsed)renderPanel();};
  $('filters-close').onclick=()=>{$('object-layers').open=false;$('layer-summary').focus();};
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!detail.hasAttribute('open')){$('object-layers').open=false;$('layer-summary').focus();}});
  document.addEventListener('click',event=>{if(!$('object-layers').contains(event.target))$('object-layers').open=false;});
  $('v-map').onclick=()=>setView('map');$('v-wall').onclick=()=>setView('wall');
  $('shuffle').onclick=()=>{shuffleSeed=1+Math.floor(Math.random()*2147483646);setView('wall',false);updateSelection();saveState();};
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
  for(const id of ['confidence-filter','source-filter','area-filter','group-duplicates','date-unknown','cluster-filter'])$(id).onchange=changeFilters;
  $('barrio-filter').onchange=()=>{drawBarrio(true);changeFilters();};
  $('cluster-colors').onchange=()=>{refreshVisible();saveState();};
  $('map-previews').onchange=()=>{observations.draw();saveState();};
  let yearTimer;
  for(const id of ['year-from','year-to'])$(id).onchange=()=>{if(!$(id).value||!$(id).checkValidity()){$(id).value=id==='year-from'?$(id).min:$(id).max;}if(Number($('year-from').value)>Number($('year-to').value))$(id==='year-from'?'year-to':'year-from').value=$(id).value;yearLabels();clearTimeout(yearTimer);yearTimer=setTimeout(changeFilters,180);};
  let searchTimer;$('object-search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(changeFilters,200);};
  $('focus-results').onclick=()=>{if(!result?.bounds)return;const [w,s,e,n]=result.bounds;setView('map');map.fitBounds([[s,w],[n,e]],{padding:[40,40],maxZoom:18});};
  $('reset-filters').onclick=()=>{
    enabled.clear();for(const id of ['graffiti','street_name_plate'])if(categories.has(id))enabled.add(id);
    document.querySelectorAll('[data-category]').forEach(input=>input.checked=enabled.has(input.dataset.category));
    $('group-duplicates').checked=true;$('cluster-colors').checked=false;$('map-previews').checked=true;shuffleSeed=0;
    $('confidence-filter').value='0.65';$('object-search').value='';$('barrio-filter').value='';$('source-filter').value='';$('area-filter').checked=false;$('date-unknown').checked=true;
    $('year-from').value=$('year-from').min;$('year-to').value=$('year-to').max;$('cluster-filter').value='';yearLabels();drawBarrio();changeFilters();
  };
  $('layers-all').onclick=()=>{for(const id of categories.keys())enabled.add(id);document.querySelectorAll('[data-category]').forEach(input=>input.checked=true);changeFilters();};
  $('layers-none').onclick=()=>{enabled.clear();document.querySelectorAll('[data-category]').forEach(input=>input.checked=false);changeFilters();};
  $('download-data').onclick=()=>downloadData();$('download-manifest').onclick=()=>downloadData(true);
  $('share').onclick=async()=>{saveState();try{await navigator.clipboard.writeText(location.href);notice(t('View link copied'));}catch(error){window.prompt(t('Copy this view link'),location.href);}};
  window.addEventListener('popstate',restoreState);
  $('snapshot-update').onclick=async()=>{
    if(!availableManifest||detail.hasAttribute('open')||queryBusy)return;
    refreshing=true;const next=availableManifest;
    $('snapshot-update').disabled=true;$('update-status').textContent=t('Updating selected layers…');
    const ok=await updateSelection(next);
    if(ok){
      for(const c of next.categories){if(!categories.has(c.id))addLayer(c);categories.set(c.id,c);}
      availableManifest=null;$('snapshot-update').hidden=true;$('update-status').textContent='';
    }else{$('update-status').textContent=t('Update failed. Your current data is unchanged; try again.');}
    refreshing=false;$('snapshot-update').disabled=false;updateCounts();
  };
  setInterval(async()=>{try{const next=await fetchJSON('/api/meta');if(next.query_snapshot!==manifest.query_snapshot){availableManifest=next;$('snapshot-update').hidden=false;}}catch(error){}},60000);
  ready=true;await restoreState();
})();
