(() => {
  'use strict';

  const header = document.querySelector('.site-header');
  const navToggle = document.querySelector('.nav-toggle');
  const navigation = document.querySelector('#site-nav');

  function setNavigationOpen(open, restoreFocus = false) {
    header.classList.toggle('nav-open', open);
    navToggle.setAttribute('aria-expanded', String(open));
    navToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    if (restoreFocus) navToggle.focus({ preventScroll: true });
  }

  navToggle.addEventListener('click', () => {
    setNavigationOpen(navToggle.getAttribute('aria-expanded') !== 'true');
  });
  navigation.addEventListener('click', (event) => {
    if (event.target.closest('a')) setNavigationOpen(false);
  });
  document.addEventListener('click', (event) => {
    if (header.classList.contains('nav-open') && !header.contains(event.target)) {
      setNavigationOpen(false);
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && header.classList.contains('nav-open')) {
      setNavigationOpen(false, true);
    }
  });

  function updateHeader() {
    header.classList.toggle('scrolled', window.scrollY > 24);
  }
  updateHeader();
  window.addEventListener('scroll', updateHeader, { passive: true });

  const tabs = [...document.querySelectorAll('.result-tabs [role="tab"]')];

  function activateTab(selected, moveFocus = false) {
    tabs.forEach((tab) => {
      const active = tab === selected;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
    });
    if (moveFocus) selected.focus();
    const tabList = selected.parentElement;
    const selectedBounds = selected.getBoundingClientRect();
    const listBounds = tabList.getBoundingClientRect();
    if (selectedBounds.right > listBounds.right) tabList.scrollLeft += selectedBounds.right - listBounds.right;
    if (selectedBounds.left < listBounds.left) tabList.scrollLeft -= listBounds.left - selectedBounds.left;
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activateTab(tab));
    tab.addEventListener('keydown', (event) => {
      let nextIndex;
      if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
      else if (event.key === 'Home') nextIndex = 0;
      else if (event.key === 'End') nextIndex = tabs.length - 1;
      else return;
      event.preventDefault();
      activateTab(tabs[nextIndex], true);
    });
  });

  const initialTab = tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') || tabs[0];
  if (initialTab) activateTab(initialTab);

  const dialog = document.querySelector('.figure-dialog');
  const dialogImage = dialog.querySelector('.dialog-image');
  const dialogViewport = dialog.querySelector('.dialog-image-viewport');
  const dialogTitle = dialog.querySelector('#figure-dialog-title');
  const dialogCaption = dialog.querySelector('.dialog-caption');
  const dialogOriginal = dialog.querySelector('.dialog-original');
  let figureTrigger = null;
  let previousOverflow = '';
  let pointerStartedOutside = false;

  document.querySelectorAll('[data-figure]').forEach((button) => {
    let pointerOrigin = null;
    let movedPointer = false;
    button.addEventListener('pointerdown', (event) => {
      pointerOrigin = { x: event.clientX, y: event.clientY };
      movedPointer = false;
    });
    function trackPointer(event) {
      if (pointerOrigin && Math.hypot(event.clientX - pointerOrigin.x, event.clientY - pointerOrigin.y) > 8) {
        movedPointer = true;
      }
    }
    button.addEventListener('pointermove', trackPointer);
    button.addEventListener('pointerup', (event) => {
      trackPointer(event);
      pointerOrigin = null;
    });
    button.addEventListener('pointercancel', () => {
      pointerOrigin = null;
      movedPointer = true;
    });
    button.addEventListener('dragstart', (event) => event.preventDefault());
    button.addEventListener('click', (event) => {
      if (movedPointer && event.detail !== 0) {
        movedPointer = false;
        return;
      }
      movedPointer = false;
      if (typeof dialog.showModal !== 'function') {
        window.open(button.dataset.figure, '_blank', 'noopener,noreferrer');
        return;
      }
      figureTrigger = button;
      dialogTitle.textContent = button.dataset.title;
      dialogCaption.textContent = button.dataset.caption;
      dialogImage.alt = button.querySelector('img').alt;
      dialogImage.src = button.dataset.figure;
      dialogOriginal.href = button.dataset.figure;
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      dialog.showModal();
      dialogViewport.scrollTo(0, 0);
    });
  });

  dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());

  function isOutsideDialog(event) {
    const bounds = dialog.getBoundingClientRect();
    return event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom;
  }

  dialog.addEventListener('pointerdown', (event) => {
    pointerStartedOutside = event.target === dialog && isOutsideDialog(event);
  });
  dialog.addEventListener('click', (event) => {
    if (pointerStartedOutside && event.target === dialog && isOutsideDialog(event)) {
      dialog.close();
    }
    pointerStartedOutside = false;
  });
  dialog.addEventListener('close', () => {
    document.body.style.overflow = previousOverflow;
    pointerStartedOutside = false;
    if (figureTrigger?.isConnected) figureTrigger.focus({ preventScroll: true });
    dialogImage.removeAttribute('src');
  });

  function copyWithSelection(text) {
    const previousFocus = document.activeElement;
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.readOnly = true;
    textarea.tabIndex = -1;
    textarea.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0;';
    document.body.append(textarea);
    textarea.select();
    textarea.setSelectionRange(0, textarea.value.length);
    let copied = false;
    try {
      copied = document.execCommand('copy');
    } finally {
      textarea.remove();
      previousFocus?.focus({ preventScroll: true });
    }
    if (!copied) throw new Error('Clipboard copy was not available.');
  }

  document.querySelectorAll('[data-copy-target]').forEach((button) => {
    const initialContent = [...button.childNodes].map((node) => node.cloneNode(true));
    const target = document.getElementById(button.dataset.copyTarget);
    const status = button.closest('.citation-block').querySelector('.copy-status');
    let resetTimer;

    button.addEventListener('click', async () => {
      window.clearTimeout(resetTimer);
      button.disabled = true;
      status.textContent = '';
      try {
        const text = target.textContent.trim();
        try {
          if (!navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable.');
          await navigator.clipboard.writeText(text);
        } catch {
          copyWithSelection(text);
        }
        button.textContent = 'Copied ✓';
        status.textContent = 'BibTeX copied to clipboard.';
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(target);
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = 'Copy manually';
        status.textContent = 'The citation is selected. Press ⌘C or Ctrl+C to copy it.';
      } finally {
        button.disabled = false;
        resetTimer = window.setTimeout(() => {
          button.replaceChildren(...initialContent.map((node) => node.cloneNode(true)));
        }, 3000);
      }
    });
  });
})();
