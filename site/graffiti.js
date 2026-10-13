'use strict';
(() => {
  const form=document.getElementById('controls'),status=document.getElementById('status'),modelStatus=document.getElementById('model-status'),button=document.getElementById('generate'),art=document.getElementById('art');
  const params=new URLSearchParams(location.search);
  if(params.has('prompt'))form.elements.prompt.value=params.get('prompt').slice(0,500);
  if(['0','0.4','0.8','1.2'].includes(params.get('strength')))form.elements.strength.value=params.get('strength');
  if(/^\d{1,10}$/.test(params.get('seed')||'')&&Number(params.get('seed'))<=2147483647)form.elements.seed.value=params.get('seed');
  let running=false,ready=false;
  async function request(path,options){const response=await fetch('/api/graffiti/'+path,{cache:'no-store',...options});const data=await response.json();if(!response.ok)throw Error(data.error||'Could not reach the image model.');return data;}
  function settings(){return {prompt:form.elements.prompt.value.trim(),seed:Number(form.elements.seed.value),strength:Number(form.elements.strength.value)};}
  function shareURL(){const url=new URL(location.href);url.search=new URLSearchParams(settings());history.replaceState(null,'',url);return url.href;}
  async function check(){
    try{const data=await request('status'),training=data.training;ready=training.stage==='trained';
      const text=ready?`Model ready. ${training.train_count||'Curated'} training images; ${training.validation_count||'separate'} held out.`:training.stage==='training'?`Fine-tuning: ${training.step} / ${training.steps} steps.`:training.stage==='encoding'?'Preparing the selected training images.':training.stage==='failed'?'Training paused after an error. The model is not ready yet.':'The first archive model is being prepared.';
      modelStatus.textContent=text;
      if(!running){button.disabled=!ready||data.busy;button.textContent=data.busy?'GPU busy…':ready?'Generate image ↗':'Model preparing…';status.textContent=text;}
    }catch(error){if(!running){button.disabled=true;status.textContent='Image service unavailable. Retrying shortly.';modelStatus.textContent='The image service is not connected.';}}
  }
  const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  form.addEventListener('submit',async event=>{event.preventDefault();if(running||!ready)return;
    running=true;button.disabled=true;button.textContent='Generating…';status.textContent='Loading the model and painting a new image. This can take a minute.';
    try{const input=settings();shareURL();let job=await request('generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)});
      const id=job.id;const start=Date.now();
      while(job.stage==='running'){if(Date.now()-start>270000)throw Error('This request took too long. Refresh to try again.');await pause(2000);job=await request('jobs/'+id);}
      if(job.stage!=='complete')throw Error(job.error||'Generation failed.');
      art.src=job.image;art.hidden=false;document.getElementById('empty').hidden=true;
      const link=document.getElementById('download');link.href=job.image;link.hidden=false;
      document.getElementById('caption').textContent=`Seed ${input.seed} · archive influence ${input.strength}`;
      status.textContent='Image ready. Try a new seed, or compare archive influence.';
    }catch(error){status.textContent=error.message;}finally{running=false;button.disabled=!ready;button.textContent='Generate image ↗';}
  });
  document.getElementById('variation').addEventListener('click',()=>{const value=new Uint32Array(1);crypto.getRandomValues(value);form.elements.seed.value=value[0]%2147483647;shareURL();});
  document.getElementById('share').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(shareURL());status.textContent='Settings copied. The link does not generate automatically.';}catch{status.textContent='Copy the page address to share these settings.';}});
  check();setInterval(()=>{if(!running)check();},15000);
})();
