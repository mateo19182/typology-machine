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
