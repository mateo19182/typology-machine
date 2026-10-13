/* Offline neighbors; request-time work is image loading and a uniform transform. */
(function(root){
  function alignment(width,height,box,frameWidth,frameHeight,size=.36,anchor=[.5,.5]){
    const objectW=(box[2]-box[0])*width,objectH=(box[3]-box[1])*height;
    if(!(objectW>0&&objectH>0&&frameWidth>0&&frameHeight>0))throw Error('Invalid object bounds');
    const scale=Math.min(frameWidth,frameHeight)*size/Math.max(objectW,objectH);
    return {width:width*scale,height:height*scale,left:anchor[0]*frameWidth-(box[0]+box[2])*.5*width*scale,top:anchor[1]*frameHeight-(box[1]+box[3])*.5*height*scale};
  }
  if(typeof module!=='undefined')module.exports={alignment};
  if(typeof document==='undefined')return;
  const $=id=>document.getElementById(id),stage=$('stage'),status=$('status');
  let graph,storage=CropStorage.create(),trail=[],position=-1,busy=false,generation=0,currentImage=null,anchor=[.5,.5];
  const cache=new Map();
  function imageFor(index){
    const url=graph.items[index].raw;
    if(cache.has(url))return cache.get(url);
    const promise=new Promise((resolve,reject)=>{
      const image=new Image();let timeout=setTimeout(()=>{image.src='';cache.delete(url);reject(Error('Street image took too long to load. Try another match.'));},45000);
      image.onload=()=>{clearTimeout(timeout);resolve(image)};
      image.onerror=()=>{clearTimeout(timeout);cache.delete(url);reject(Error('This street image is unavailable. Try another match.'));};
      image.src=url;
    });
    cache.set(url,promise);
    while(cache.size>6)cache.delete(cache.keys().next().value);
    return promise;
  }
  function index(){return trail[position]}
  function matches(){
    const neighbors=graph.neighbors[index()]||[];
    const fresh=neighbors.filter(n=>!trail.slice(0,position+1).includes(n));
    return fresh.length?fresh:neighbors;
  }
  function place(image,row){
    const a=alignment(image.naturalWidth,image.naturalHeight,row.bx,stage.clientWidth,stage.clientHeight,Number($('anchor-size').value)/100,anchor);
    Object.assign(image.style,{width:a.width+'px',height:a.height+'px',left:a.left+'px',top:a.top+'px'});
    $('anchor').style.left=anchor[0]*100+'%';$('anchor').style.top=anchor[1]*100+'%';
  }
  function save(){
    const p=new URLSearchParams({path:trail.map(n=>graph.items[n].id).join('|'),at:position,size:$('anchor-size').value,ax:anchor[0].toFixed(3),ay:anchor[1].toFixed(3)});
    history.replaceState(null,'','#'+p.toString());
  }
  function controls(){
    $('back').disabled=busy||position<=0;$('next').disabled=busy||position<0||!matches().length;
    $('restart').disabled=busy||!graph;$('share').disabled=busy||position<0;$('category').disabled=busy;
    $('position').textContent=position<0?'':`Street ${position+1} / ${trail.length}`;
    document.querySelectorAll('#alternatives button').forEach(b=>b.disabled=busy);
  }
  function alternatives(){
    $('alternatives').replaceChildren();
    for(const n of matches().slice(0,6)){
      const row=graph.items[n],button=document.createElement('button'),im=new Image(),caption=document.createElement('span');
      im.alt=row.text||row.category.replaceAll('_',' ');im.loading='lazy';storage.image(im,row.crop,()=>{im.alt='Crop unavailable'});
      caption.textContent=row.barrio||row.source;button.append(im,caption);button.title='Continue with this object';button.onclick=()=>show(n,'append');$('alternatives').append(button);
    }
  }
  async function show(n,mode='append',targetPosition){
    if(busy||!Number.isInteger(n)||!graph.items[n])return;
    busy=true;const turn=++generation;controls();status.textContent='Loading the next real street image…';
    try{
      const loaded=await imageFor(n);
      if(turn!==generation)return;
      // Each layer owns a separate node; cached images never move between layers.
      const image=loaded.cloneNode();image.alt=`${graph.items[n].category.replaceAll('_',' ')} in ${graph.items[n].barrio||'A Coruña'}`;
      // cloneNode retains decoded src but may not expose dimensions until loaded.
      if(!image.complete)await image.decode();
      place(image,graph.items[n]);$('images').append(image);
      const old=currentImage;currentImage=image;
      requestAnimationFrame(()=>requestAnimationFrame(()=>{image.classList.add('shown');if(old)old.classList.remove('shown')}));
      setTimeout(()=>{if(old&&old!==currentImage)old.remove()},450);
      if(mode==='reset'){trail=[n];position=0}
      else if(mode==='back'||mode==='restore'){position=targetPosition}
      else{trail=trail.slice(0,position+1);trail.push(n);if(trail.length>40)trail.shift();position=trail.length-1}
      const row=graph.items[n];$('category').value=row.category;
      $('stage-empty').hidden=true;$('caption').textContent=[row.category.replaceAll('_',' '),row.barrio||'A Coruña',row.d?String(row.d).slice(0,10):'Date unknown',row.source==='apple'?'Apple Look Around':'Google Street View'].join(' · ');
      $('source').hidden=!row.provider_url;
      if(row.provider_url){try{const url=new URL(row.provider_url);if(url.protocol==='https:')$('source').href=url.href;else $('source').hidden=true}catch{$('source').hidden=true}}
      $('object-link').hidden=false;$('object-link').href='index.html#'+new URLSearchParams({layers:row.category,object:row.object_id,confidence:'0.65',lat:row.la,lon:row.lo,z:18});
      alternatives();save();status.textContent='Matching shape anchored. Choose Next or a crop below.';
      // Prefetch only one context to avoid flooding the on-demand decoder.
      const next=matches()[0];if(next!==undefined)imageFor(next).catch(()=>{});
    }catch(error){status.textContent=error.message;if(!currentImage)$('stage-empty').textContent='Image unavailable. Choose a new starting point.'}
    finally{busy=false;controls()}
  }
  function restart(){if(!graph)return;const category=$('category').value,choices=graph.items.map((r,n)=>r.category===category?n:-1).filter(n=>n>=0&&n!==index());if(choices.length)show(choices[Math.floor(Math.random()*choices.length)],'reset')}
  $('next').onclick=()=>{if(position+1<trail.length)show(trail[position+1],'back',position+1);else if(matches().length)show(matches()[0])};
  $('back').onclick=()=>{if(position>0)show(trail[position-1],'back',position-1)};
  $('restart').onclick=restart;$('category').onchange=restart;
  $('anchor-size').oninput=()=>{if(currentImage){place(currentImage,graph.items[index()]);save()}};
  $('outline').onchange=()=>{$('anchor').hidden=!$('outline').checked};
  stage.onclick=event=>{if(!currentImage)return;const r=stage.getBoundingClientRect();anchor=[Math.min(.85,Math.max(.15,(event.clientX-r.left)/r.width)),Math.min(.85,Math.max(.15,(event.clientY-r.top)/r.height))];place(currentImage,graph.items[index()]);save()};
  stage.onkeydown=event=>{if(event.key==='ArrowRight'){event.preventDefault();$('next').click()}if(event.key==='ArrowLeft'){event.preventDefault();$('back').click()}};
  new ResizeObserver(()=>{if(currentImage)place(currentImage,graph.items[index()])}).observe(stage);
  $('share').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);status.textContent='Journey link copied.'}catch{window.prompt('Copy journey link',location.href)}};
  async function init(){
    try{
      const [response,config]=await Promise.all([fetch('wander-data.json'),fetch('crop-storage.json').then(r=>r.ok?r.json():{}).catch(()=>({}))]);
      if(!response.ok)throw Error('The visual match collection is not available yet.');
      graph=await response.json();storage=CropStorage.create(config);
      if(!graph.items?.length||graph.neighbors?.length!==graph.items.length)throw Error('The visual match collection is empty or incomplete.');
      $('coverage').textContent=`This first collection contains ${graph.items.length.toLocaleString()} objects sampled across ${Object.keys(graph.categories).length} categories, at confidence 0.75 or higher. Matches were prepared on ${graph.built_at.slice(0,10)}; it is a sample of the archive.`;
      const p=new URLSearchParams(location.hash.slice(1)),lookup=new Map(graph.items.map((r,i)=>[r.id,i]));
      trail=(p.get('path')||'').split('|').slice(0,40).map(id=>lookup.get(id)).filter(n=>n!==undefined);
      const size=Number(p.get('size'));if(size>=18&&size<=65)$('anchor-size').value=size;
      for(const [i,k]of ['ax','ay'].entries()){const v=Number(p.get(k));if(p.has(k)&&v>=.15&&v<=.85)anchor[i]=v}
      controls();
      if(trail.length){const at=Math.max(0,Math.min(trail.length-1,parseInt(p.get('at'),10)||0));await show(trail[at],'restore',at)}else restart();
    }catch(error){status.textContent=error.message;$('stage-empty').textContent='Could not load this experiment. Reload to retry.'}
  }
  init();
})(typeof window==='undefined'?globalThis:window);
