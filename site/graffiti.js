'use strict';
(() => {
  const form=document.getElementById('controls'),status=document.getElementById('status'),button=document.getElementById('generate'),art=document.getElementById('art');
  const params=new URLSearchParams(location.search);
  if(params.has('prompt'))form.elements.prompt.value=params.get('prompt').slice(0,500);
  if(params.get('strength'))setStrength(params.get('strength'));
  if(/^\d{1,10}$/.test(params.get('seed')||'')&&Number(params.get('seed'))<=2147483647)form.elements.seed.value=params.get('seed');
  let running=false,ready=false,hasOutcome=false,generationSteps=4;
  function setStrength(value){
    const number=Number(value);if(!Number.isFinite(number)||number<0||number>1.2)return;
    const select=form.elements.strength;
    if(!Array.from(select.options).some(option=>Number(option.value)===number)){
      const option=document.createElement('option');option.value=String(number);option.textContent=`Custom · ${number}`;select.append(option);
    }
    select.value=String(number);
  }
  const progressPanel=document.getElementById('generation-progress'),progressBar=document.getElementById('drawing-progress');
  function showProgress(job){
    const progress=job.progress||{phase:'loading',completed_steps:0,total_steps:generationSteps};
    progressBar.max=progress.total_steps;
    if(['generating','finishing','complete'].includes(progress.phase))progressBar.value=progress.completed_steps;
    else progressBar.removeAttribute('value');
  }
  async function request(path,options){const response=await fetch('/api/graffiti/'+path,{cache:'no-store',...options});const data=await response.json();if(!response.ok)throw Error(data.error||'Could not reach the image model.');return data;}
  function settings(){return {prompt:form.elements.prompt.value.trim(),seed:Number(form.elements.seed.value),strength:Number(form.elements.strength.value)};}
  function shareURL(){const url=new URL(location.href);url.search=new URLSearchParams(settings());history.replaceState(null,'',url);return url.href;}
  async function check(){
    try{const data=await request('status'),training=data.training,loading=data.inference_worker?.stage==='loading';generationSteps=data.steps;ready=training.stage==='trained'&&!data.paused&&!loading;
      const text=ready?'':training.stage==='training'?`Fine-tuning: ${training.step} / ${training.steps} steps.`:training.stage==='encoding'?'Preparing the selected training images.':training.stage==='failed'?'Training paused after an error. The model is not ready yet.':'The archive model is being prepared.';
      const message=loading?'Preparing the image model…':data.paused&&training.stage==='trained'?'Checking the trained model before opening the generator.':text;
      if(!running){button.disabled=!ready||data.busy;button.textContent=data.busy?'GPU busy…':ready?'Generate image ↗':'Model preparing…';if(!hasOutcome)status.textContent=message;}
    }catch(error){if(!running){button.disabled=true;status.textContent='Image service unavailable. Retrying shortly.';}}
  }
  const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  form.addEventListener('submit',async event=>{event.preventDefault();if(running||!ready)return;
    running=true;button.disabled=true;button.textContent='Generating…';status.textContent='Making your image.';
    progressPanel.hidden=false;art.setAttribute('aria-busy','true');showProgress({});
    try{const input=settings();shareURL();let job=await request('generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)});
      const id=job.id;const start=Date.now();
      showProgress(job);
      while(job.stage==='running'){if(Date.now()-start>630000)throw Error('This request took too long. Refresh to try again.');await pause(1000);job=await request('jobs/'+id);showProgress(job);}
      if(job.stage!=='complete')throw Error(job.error||'Generation failed.');
      art.src=job.image;art.hidden=false;document.getElementById('empty').hidden=true;
      const link=document.getElementById('download');link.href=job.image;link.hidden=false;
      document.getElementById('caption').textContent=`Seed ${input.seed} · archive influence ${input.strength}`;
      status.textContent='Image ready. Keep the seed fixed and compare archive influence.';
      loadLibrary();
    }catch(error){status.textContent=error.message;}finally{progressPanel.hidden=true;art.removeAttribute('aria-busy');running=false;hasOutcome=true;button.disabled=!ready;button.textContent='Generate image ↗';check();}
  });
  document.getElementById('variation').addEventListener('click',()=>{const value=new Uint32Array(1);crypto.getRandomValues(value);form.elements.seed.value=value[0]%2147483647;shareURL();});
  document.getElementById('share').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(shareURL());status.textContent='Settings copied. The link does not generate automatically.';}catch{status.textContent='Copy the page address to share these settings.';}});
  const grid=document.getElementById('library-grid'),more=document.getElementById('library-more'),libraryEmpty=document.getElementById('library-empty'),viewer=document.getElementById('library-viewer');
  let libraryItems=[],visibleItems=18,librarySignature=null,selectedImage=null;
  function openImage(item){
    selectedImage=item;
    document.getElementById('viewer-image').src=item.image;
    document.getElementById('viewer-download').href=item.image;
    const prompt=document.getElementById('viewer-prompt'),meta=document.getElementById('viewer-meta'),reuse=document.getElementById('viewer-reuse');
    const hasSettings=typeof item.prompt==='string'&&Number.isInteger(item.seed)&&Number.isFinite(item.strength);
    prompt.textContent=hasSettings?item.prompt:'';prompt.hidden=!hasSettings;
    meta.textContent=hasSettings?`Seed ${item.seed} · archive influence ${item.strength}`:'';meta.hidden=!hasSettings;
    reuse.hidden=!hasSettings;
    viewer.showModal();
  }
  function renderLibrary(){
    const cards=libraryItems.slice(0,visibleItems).map((item,index)=>{
      const card=document.createElement('button');card.type='button';card.className='library-item';card.setAttribute('aria-label',`View generation ${index+1}`);
      const image=document.createElement('img');image.src=item.thumbnail;image.alt='';image.width=256;image.height=256;image.loading='lazy';image.decoding='async';
      image.addEventListener('error',()=>{if(image.getAttribute('src')!==item.image)image.src=item.image;});
      card.append(image);card.addEventListener('click',()=>openImage(item));return card;
    });
    grid.replaceChildren(...cards);more.hidden=visibleItems>=libraryItems.length;
    document.getElementById('library-count').textContent=libraryItems.length||'';
    libraryEmpty.hidden=libraryItems.length>0;libraryEmpty.textContent='No generations yet.';
  }
  async function loadLibrary(){
    try{const data=await request('library'),signature=JSON.stringify(data.images);
      if(signature!==librarySignature){libraryItems=data.images;librarySignature=signature;renderLibrary();}
      if(!running&&art.hidden&&libraryItems.length){
        const latest=libraryItems[0];art.src=latest.image;art.hidden=false;document.getElementById('empty').hidden=true;
        const download=document.getElementById('download');download.href=latest.image;download.hidden=false;
        document.getElementById('caption').textContent=Number.isInteger(latest.seed)?`Latest generation · seed ${latest.seed} · archive influence ${latest.strength}`:'Latest generation';
      }
    }catch(error){if(!libraryItems.length){libraryEmpty.hidden=false;libraryEmpty.textContent='Library unavailable.';}}
  }
  more.addEventListener('click',()=>{visibleItems+=18;renderLibrary();});
  document.getElementById('viewer-close').addEventListener('click',()=>viewer.close());
  viewer.addEventListener('click',event=>{
    const box=viewer.getBoundingClientRect();
    if(event.target===viewer&&(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom))viewer.close();
  });
  document.getElementById('viewer-reuse').addEventListener('click',()=>{
    form.elements.prompt.value=selectedImage.prompt;form.elements.seed.value=selectedImage.seed;setStrength(selectedImage.strength);shareURL();viewer.close();form.elements.prompt.focus();
  });
  check();loadLibrary();setInterval(()=>{if(!running){check();loadLibrary();}},15000);
})();
