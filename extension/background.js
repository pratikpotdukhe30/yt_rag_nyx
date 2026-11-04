// Clicking the toolbar icon opens the side panel. Service worker = runs only when needed.
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch((err) => console.error(err));