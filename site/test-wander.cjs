const test=require('node:test');const assert=require('node:assert/strict');const {alignment}=require('./wander.js');
test('different source sizes and boxes keep centre and object long side fixed',()=>{
 for(const [w,h,b]of [[1600,900,[.1,.2,.4,.6]],[900,1600,[.6,.7,.9,.8]]]){
  const a=alignment(w,h,b,1000,600,.36,[.4,.7]);
  assert.ok(Math.abs(a.left+(b[0]+b[2])*.5*a.width-400)<1e-8);
  assert.ok(Math.abs(a.top+(b[1]+b[3])*.5*a.height-420)<1e-8);
  assert.ok(Math.abs(Math.max((b[2]-b[0])*a.width,(b[3]-b[1])*a.height)-216)<1e-8);
 }
});
test('rejects missing or inverted object area',()=>assert.throws(()=>alignment(100,100,[.5,0,.4,1],600,400)));
const {composePose,identity,validPose}=require('./wander.js');
test('composes canonical crop transforms in source coordinates',()=>{
 const p={angle:10,scale:1.1,x:.02,y:-.01};const pair={angle:-5,scale:.9,x:.03,y:.04};const q=composePose(p,pair),r=10*Math.PI/180;
 assert.equal(q.angle,5);assert.ok(Math.abs(q.scale-.99)<1e-12);
 assert.ok(Math.abs(q.x-(.02+1.1*(Math.cos(r)*.03-Math.sin(r)*.04)))<1e-12);
 assert.ok(Math.abs(q.y-(-.01+1.1*(Math.sin(r)*.03+Math.cos(r)*.04)))<1e-12);
 assert.equal(q.aligned,true);
});
test('falls back to centred identity instead of accumulating angle, scale or position drift',()=>{
 for(const [a,b]of [[{angle:25,scale:1,x:0,y:0},{angle:10,scale:1,x:0,y:0}],[{angle:0,scale:1.3,x:0,y:0},{angle:0,scale:1.2,x:0,y:0}],[{angle:0,scale:1,x:.2,y:0},{angle:0,scale:1,x:.1,y:0}],[identity(),{angle:45,scale:1,x:0,y:0}],[identity(),null]])assert.deepEqual(composePose(a,b),identity());
 assert.equal(validPose({angle:NaN,scale:1,x:0,y:0}),false);
});
test('rotated photo maps canonical object coordinates to the intended screen transform',()=>{
 const width=1600,height=900,box=[.1,.2,.4,.6],pose={angle:15,scale:1.2,x:.05,y:-.08};
 const a=alignment(width,height,box,1000,600,.36,[.4,.7],pose),r=pose.angle*Math.PI/180;
 const cx=(box[0]+box[2])/2*a.width,cy=(box[1]+box[3])/2*a.height;
 assert.ok(Math.abs(a.left+Math.cos(r)*cx-Math.sin(r)*cy-(400+216*pose.x))<1e-8);
 assert.ok(Math.abs(a.top+Math.sin(r)*cx+Math.cos(r)*cy-(420+216*pose.y))<1e-8);
 assert.equal(a.angle,15);
});
