const CACHE_NAME = 'halal-scanner-v2'; // Changed to v2 to force update
const urlsToCache = [
  '/',
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  // Cache the external library so icons work offline
  'https://unpkg.com/feather-icons',
  // Cache the font definition
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap'
];

// 1. INSTALL: Cache all static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Opened cache');
        return cache.addAll(urlsToCache);
      })
  );
});

// 2. ACTIVATE: Clean up old caches (Delete 'halal-scanner-v1')
self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// 3. FETCH: Network-First for API, Cache-First for Assets
self.addEventListener('fetch', event => {
  // Check if the request is for the API
  if (event.request.url.includes('/api/')) {
    // For API calls, go straight to network. Do not cache.
    event.respondWith(fetch(event.request));
    return;
  }

  // For everything else (HTML, CSS, JS, Images), try Cache first
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Cache hit - return response
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});