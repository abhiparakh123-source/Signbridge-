const CACHE="signbridge-shell-v3";
const SHELL=["/","/manifest.webmanifest","/signbridge-mark.svg"];

self.addEventListener("install",event=>{
  event.waitUntil((async()=>{
    const cache=await caches.open(CACHE);
    await cache.addAll(SHELL);
    await self.skipWaiting();
  })());
});

self.addEventListener("activate",event=>{
  event.waitUntil((async()=>{
    const keys=await caches.keys();
    await Promise.all(keys.filter(key=>key.startsWith("signbridge-")&&key!==CACHE).map(key=>caches.delete(key)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch",event=>{
  const request=event.request;
  if(request.method!=="GET")return;
  const url=new URL(request.url);
  if(url.origin!==self.location.origin)return;

  if(request.mode==="navigate"){
    event.respondWith((async()=>{
      const cache=await caches.open(CACHE);
      try{
        const response=await fetch(request);
        if(response&&response.ok)await cache.put("/",response.clone());
        return response;
      }catch{
        return (await cache.match(request))||(await cache.match("/"));
      }
    })());
    return;
  }

  // Cache successful same-origin app assets after the browser fetches them.
  // This makes Vite's hashed JS/CSS bundles available for later offline launches.
  event.respondWith((async()=>{
    const cache=await caches.open(CACHE);
    const cached=await cache.match(request);
    if(cached)return cached;
    try{
      const response=await fetch(request);
      if(response&&response.ok&&response.type==="basic")await cache.put(request,response.clone());
      return response;
    }catch{
      return cached||Response.error();
    }
  })());
});
