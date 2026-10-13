/* Only public crop paths may use remote storage. Full street images stay local. */
(function(root){
  function create(config={}){
    let base=null;
    try{
      const value=new URL(config.base_url);
      if(value.protocol==='https:'&&!value.username&&!value.password&&!value.search&&!value.hash)base=value.href.replace(/\/$/,'');
    }catch(error){}
    function url(path){
      if(!base||typeof path!=='string')return path;
      const local=path.startsWith('/')?path:'/'+path;
      if(!/^\/(img|proposal-assets)\/[A-Za-z0-9_./-]+\.jpg$/.test(local)||local.split('/').some(part=>part==='.'||part==='..'))return path;
      return base+local;
    }
    function image(element,path,onFailure){
      const remote=url(path);
      let fallback=false;
      element.onerror=()=>{
        if(!fallback&&remote!==path){fallback=true;element.src=path;return;}
        if(onFailure)onFailure();
      };
      element.src=remote;
    }
    return {url,image};
  }
  root.CropStorage={create};
  if(typeof module!=='undefined')module.exports={create};
})(typeof window==='undefined'?globalThis:window);
