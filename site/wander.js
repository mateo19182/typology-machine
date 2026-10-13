/* Offline neighbors; request-time work is image loading and a uniform transform. */
(function(root){
  const identity=()=>({angle:0,scale:1,x:0,y:0,aligned:false});
  function validPose(p){return p&&['angle','scale','x','y'].every(k=>Number.isFinite(p[k]))&&Math.abs(p.angle)<=30&&p.scale>=.75&&p.scale<=1.33&&Math.hypot(p.x,p.y)<=.25;}
  function composePose(previous,pair){
    if(!validPose(previous)||!pair||!['angle','scale','x','y'].every(k=>Number.isFinite(pair[k]))||Math.abs(pair.angle)>20||pair.scale<.8||pair.scale>1.25||Math.abs(pair.x)>.15||Math.abs(pair.y)>.15)return identity();
    const radians=previous.angle*Math.PI/180,c=Math.cos(radians),s=Math.sin(radians);
    const pose={angle:previous.angle+pair.angle,scale:previous.scale*pair.scale,x:previous.x+previous.scale*(c*pair.x-s*pair.y),y:previous.y+previous.scale*(s*pair.x+c*pair.y),aligned:true};
    return validPose(pose)?pose:identity();
  }
  function alignment(width,height,box,frameWidth,frameHeight,size=.36,anchor=[.5,.5],pose=identity()){
    const objectW=(box[2]-box[0])*width,objectH=(box[3]-box[1])*height;
    if(!(objectW>0&&objectH>0&&frameWidth>0&&frameHeight>0)||!validPose(pose))throw Error('Invalid object bounds or alignment');
    const unit=Math.min(frameWidth,frameHeight)*size,scale=unit*pose.scale/Math.max(objectW,objectH);
    const centerX=(box[0]+box[2])*.5*width*scale,centerY=(box[1]+box[3])*.5*height*scale;
    const radians=pose.angle*Math.PI/180,c=Math.cos(radians),s=Math.sin(radians);
    return {width:width*scale,height:height*scale,left:anchor[0]*frameWidth+unit*pose.x-(c*centerX-s*centerY),top:anchor[1]*frameHeight+unit*pose.y-(s*centerX+c*centerY),angle:pose.angle};
  }
  if(typeof module!=='undefined')module.exports={alignment,composePose,validPose,identity};
  if(typeof document==='undefined')return;
  const $=id=>document.getElementById(id),stage=$('stage'),status=$('status');
  let graph,storage=CropStorage.create(),trail=[],poses=[],position=-1,busy=false,generation=0,currentImage=null,anchor=[.5,.5];
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
    const table=$('match-mode').value==='color'&&graph.color_neighbors?graph.color_neighbors:graph.neighbors;
    const neighbors=table[index()]||[];
    const fresh=neighbors.filter(n=>!trail.slice(0,position+1).includes(n));
    return fresh.length?fresh:neighbors;
  }
  function place(image,row,pose=poses[position]||identity()){
    const a=alignment(image.naturalWidth,image.naturalHeight,row.bx,stage.clientWidth,stage.clientHeight,Number($('anchor-size').value)/100,anchor,pose);
    Object.assign(image.style,{width:a.width+'px',height:a.height+'px',left:a.left+'px',top:a.top+'px',transform:`rotate(${a.angle}deg)`,transformOrigin:'0 0'});
    $('anchor').style.left=anchor[0]*100+'%';$('anchor').style.top=anchor[1]*100+'%';
  }
  function save(){
    const p=new URLSearchParams({path:trail.map(n=>graph.items[n].id).join('|'),at:position,size:$('anchor-size').value,ax:anchor[0].toFixed(3),ay:anchor[1].toFixed(3),mode:$('match-mode').value,poses:JSON.stringify(poses)});
    history.replaceState(null,'','#'+p.toString());
  }
  function controls(){
    $('back').disabled=busy||position<=0;$('next').disabled=busy||position<0||(position+1>=trail.length&&!matches().length);
    $('restart').disabled=busy||!graph;$('share').disabled=busy||position<0;$('category').disabled=busy;$('match-mode').disabled=busy;
    $('position').textContent=position<0?'':`Street ${position+1} / ${trail.length}`;
    document.querySelectorAll('#alternatives button').forEach(b=>b.disabled=busy);
  }
  function alternatives(){
    $('alternatives').replaceChildren();
    if(!matches().length){const message=document.createElement('p');message.textContent='No strong matches here. Go back, change matching mode or choose a new starting point.';$('alternatives').append(message);}
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
      const pose=mode==='reset'?identity():mode==='back'||mode==='restore'?(poses[targetPosition]||identity()):composePose(poses[position],graph.alignments?.[`${index()}:${n}`]);
      place(image,graph.items[n],pose);$('images').append(image);
      const old=currentImage;currentImage=image;
      requestAnimationFrame(()=>requestAnimationFrame(()=>{image.classList.add('shown');if(old)old.classList.remove('shown')}));
      setTimeout(()=>{if(old&&old!==currentImage)old.remove()},450);
      if(mode==='reset'){trail=[n];poses=[pose];position=0}
      else if(mode==='back'||mode==='restore'){position=targetPosition}
      else{trail=trail.slice(0,position+1);poses=poses.slice(0,position+1);trail.push(n);poses.push(pose);if(trail.length>40){trail.shift();poses.shift();}position=trail.length-1}
      const row=graph.items[n];$('category').value=row.category;
      $('stage-empty').hidden=true;$('caption').textContent=[row.category.replaceAll('_',' '),row.barrio||'A Coruña',row.d?String(row.d).slice(0,10):'Date unknown',row.source==='apple'?'Apple Look Around':'Google Street View'].join(' · ');
      $('source').hidden=!row.provider_url;
      if(row.provider_url){try{const url=new URL(row.provider_url);if(url.protocol==='https:')$('source').href=url.href;else $('source').hidden=true}catch{$('source').hidden=true}}
      $('object-link').hidden=false;$('object-link').href='index.html#'+new URLSearchParams({layers:row.category,object:row.object_id,confidence:'0.65',lat:row.la,lon:row.lo,z:18});
      alternatives();save();status.textContent=pose.aligned?'Visual landmarks aligned. Choose Next or a crop below.':'Object box centred. Choose Next or a crop below.';
      // Prefetch only one context to avoid flooding the on-demand decoder.
      const next=matches()[0];if(next!==undefined)imageFor(next).catch(()=>{});
    }catch(error){status.textContent=error.message;if(!currentImage)$('stage-empty').textContent='Image unavailable. Choose a new starting point.'}
    finally{busy=false;controls()}
  }
  function restart(){if(!graph)return;const category=$('category').value,table=$('match-mode').value==='color'&&graph.color_neighbors?graph.color_neighbors:graph.neighbors;let choices=graph.items.map((r,n)=>r.category===category&&table[n]?.length?n:-1).filter(n=>n>=0);if(choices.length>1)choices=choices.filter(n=>n!==index());if(choices.length)show(choices[Math.floor(Math.random()*choices.length)],'reset');else{status.textContent='No strong matches for this category and mode. Try another category or matching mode.';controls();}}
  $('next').onclick=()=>{if(position+1<trail.length)show(trail[position+1],'back',position+1);else if(matches().length)show(matches()[0])};
  $('back').onclick=()=>{if(position>0)show(trail[position-1],'back',position-1)};
  $('restart').onclick=restart;$('category').onchange=restart;
  $('match-mode').onchange=()=>{if(graph&&position<0){restart();return;}if(graph&&position>=0){alternatives();save();controls();status.textContent=$('match-mode').value==='color'?'Following shape and colour.':'Following shape.';}};
  $('anchor-size').oninput=()=>{if(currentImage){place(currentImage,graph.items[index()]);save()}};
  $('outline').onchange=()=>{$('anchor').hidden=!$('outline').checked};
  stage.onclick=event=>{if(!currentImage)return;const r=stage.getBoundingClientRect();anchor=[Math.min(.85,Math.max(.15,(event.clientX-r.left)/r.width)),Math.min(.85,Math.max(.15,(event.clientY-r.top)/r.height))];place(currentImage,graph.items[index()]);save()};
  document.addEventListener('keydown',event=>{
    if(event.defaultPrevented||event.repeat||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;
    if(event.target.closest('input,select,textarea,[contenteditable]:not([contenteditable="false"]),[role="textbox"]'))return;
    const button=event.key==='ArrowRight'?$('next'):event.key==='ArrowLeft'?$('back'):null;
    if(button){event.preventDefault();if(!button.disabled)button.click()}
  });
  new ResizeObserver(()=>{if(currentImage)place(currentImage,graph.items[index()])}).observe(stage);
  $('share').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);status.textContent='Journey link copied.'}catch{window.prompt('Copy journey link',location.href)}};
  async function init(){
    try{
      const [response,config]=await Promise.all([fetch('wander-data.json'),fetch('crop-storage.json').then(r=>r.ok?r.json():{}).catch(()=>({}))]);
      if(!response.ok)throw Error('The visual match collection is not available yet.');
      graph=await response.json();storage=CropStorage.create(config);
      if(!graph.items?.length||graph.neighbors?.length!==graph.items.length)throw Error('The visual match collection is empty or incomplete.');
      const confidence=graph.min_confidence??graph.confidence??graph.minimum_confidence;
      $('coverage').textContent=`This collection contains ${graph.items.length.toLocaleString()} objects sampled across ${Object.keys(graph.categories).length} categories${typeof confidence==='number'?`, at confidence ${confidence} or higher`:''}. Matches were prepared${graph.built_at?' on '+graph.built_at.slice(0,10):' offline'}. ${graph.color_neighbors?'Shape + colour uses visual resemblance and colour agreement. Shape follows visual resemblance alone. ':''}${graph.alignments?'Landmark fits guide rotation and scale when reliable. ':''}It is a sample of the archive; some objects have no strong matches.`;
      const p=new URLSearchParams(location.hash.slice(1));
      if(!graph.color_neighbors||graph.color_neighbors.length!==graph.items.length){$('match-mode').value='shape';$('match-mode').querySelector('[value="color"]').disabled=true;}else $('match-mode').value=p.get('mode')==='shape'?'shape':'color';
      const lookup=new Map(graph.items.map((r,i)=>[r.id,i]));
      let savedPoses=[];try{const parsed=JSON.parse(p.get('poses')||'[]');if(Array.isArray(parsed))savedPoses=parsed.slice(0,40);}catch{}
      trail=[];poses=[];for(const [step,id] of (p.get('path')||'').split('|').slice(0,40).entries()){const n=lookup.get(id);if(n===undefined)continue;trail.push(n);const pose=savedPoses[step];poses.push(validPose(pose)?{angle:pose.angle,scale:pose.scale,x:pose.x,y:pose.y,aligned:pose.aligned===true}:identity());}
      const size=Number(p.get('size'));if(size>=18&&size<=65)$('anchor-size').value=size;
      for(const [i,k]of ['ax','ay'].entries()){const v=Number(p.get(k));if(p.has(k)&&v>=.15&&v<=.85)anchor[i]=v}
      controls();
      if(trail.length){const at=Math.max(0,Math.min(trail.length-1,parseInt(p.get('at'),10)||0));await show(trail[at],'restore',at)}else restart();
    }catch(error){status.textContent=error.message;$('stage-empty').textContent='Could not load this experiment. Reload to retry.'}
  }
  init();
})(typeof window==='undefined'?globalThis:window);
