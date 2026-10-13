// Run against the local Caddy/API: NODE_PATH=<playwright node_modules> node site/test-mobile-map.cjs
// Repository UI assets override the running site's copies. LIVE=1 checks deployed assets.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');
(async () => {
  const browser = await chromium.launch({headless:true});
  try {
    const origin = process.env.MAP_URL || 'http://127.0.0.1:8790';
    for (const width of (process.env.WIDTHS || '390').split(',').map(Number)) {
      const page = await browser.newPage({viewport:{width,height:844},isMobile:true,hasTouch:true});
      if (!process.env.LIVE) {
        await page.route(origin+'/', route => route.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(__dirname,'index.html'))}));
        await page.route('**/app.js*', route => route.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'app.js'))}));
      }
      // Give each real query enough duration to expose busy/ready layout oscillation.
      await page.route('**/api/query?*', async route => {
        const response = await route.fetch();
        await new Promise(resolve => setTimeout(resolve,100));
        await route.fulfill({response});
      });
      let queries=0;
      page.on('request',request=>{if(new URL(request.url()).pathname==='/api/query')queries++;});
      const first=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/query'&&response.ok());
      await page.goto(origin+'/',{waitUntil:'domcontentloaded'});
      await first;
      await page.waitForTimeout(1000);
      const before=queries;
      const samples=await page.evaluate(async()=>{
        const values=[];
        for(let i=0;i<60;i++){
          const r=document.getElementById('map').getBoundingClientRect();
          values.push({top:r.top,height:r.height,count:document.getElementById('count').textContent});
          await new Promise(resolve=>setTimeout(resolve,50));
        }
        return values;
      });
      const shifts=samples.filter((sample,i)=>i&&(Math.abs(sample.top-samples[i-1].top)>1||Math.abs(sample.height-samples[i-1].height)>1));
      console.log(JSON.stringify({width,idleQueries:queries-before,idleLayoutShifts:shifts.length,first:samples[0],last:samples.at(-1)}));
      assert.equal(shifts.length,0,'Untouched mobile map must not move while its loading status changes');
      assert.equal(queries-before,0,'Untouched mobile map must not repeatedly refetch its viewport');
      // User-triggered requests must still work after the idle loop is fixed.
      const next=page.waitForResponse(response=>new URL(response.url()).pathname==='/api/query'&&response.ok());
      await page.locator('.leaflet-control-zoom-in').click();
      await next;
      await page.close();
    }
  } finally {for(const context of browser.contexts())for(const page of context.pages())await page.unrouteAll({behavior:'ignoreErrors'});await browser.close();}
})().catch(error=>{console.error(error.message);process.exitCode=1;});
