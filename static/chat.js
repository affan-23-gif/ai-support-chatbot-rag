// Shared chat logic for the full-page chat (/) and the store widget (/store).
// Usage: SupportChat.mount({ log, form, input, send, suggest })
(function () {
  const LABELS = { unavailable: 'Temporarily unavailable', not_found: 'Not in our docs', handoff: 'Passed to our team',
                   order_status: 'Order status', ask_order_details: 'Order lookup', order_not_verified: 'Not verified' };

  // Keep the same conversation across page reloads (the server stores it for 24 h).
  function newId() { const a = new Uint8Array(16); crypto.getRandomValues(a); return Array.from(a, b => b.toString(16).padStart(2, '0')).join(''); }
  let session;
  try { session = sessionStorage.getItem('sb_session') || newId(); sessionStorage.setItem('sb_session', session); } catch { session = newId(); }

  // Render **bold** safely: only text nodes, never innerHTML.
  function setText(el, text) {
    String(text).split(/(\*\*[^*]+\*\*)/).forEach(part => {
      if (part.startsWith('**') && part.endsWith('**')) { const b = document.createElement('strong'); b.textContent = part.slice(2, -2); el.appendChild(b); }
      else el.appendChild(document.createTextNode(part));
    });
  }

  function mount({ log, form, input, send, suggest }) {
    const bubble = cls => { const d = document.createElement('div'); d.className = 'msg ' + cls; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; };

    function botMessage(r) {
      const d = bubble('bot ' + r.action);
      if (LABELS[r.action]) {
        const b = document.createElement('div'); b.className = 'badge';
        b.textContent = LABELS[r.action] + (r.ticket_id ? ' · #' + r.ticket_id : '');
        d.append(b, document.createElement('br'));
      }
      setText(d, r.answer);
      if (r.citations && r.citations.length) {
        const c = document.createElement('div'); c.className = 'cites';
        r.citations.forEach(ct => {
          const det = document.createElement('details'), s = document.createElement('summary'), p = document.createElement('p');
          s.textContent = `Source [${ct.n}]: ${ct.source}` + (ct.page ? ` · page ${ct.page}` : (ct.section ? ` · ${ct.section.split(' › ').pop()}` : ''));
          p.textContent = ct.snippet + (ct.snippet.length >= 220 ? '…' : '');
          det.append(s, p); c.appendChild(det);
        });
        d.appendChild(c);
      }
      log.scrollTop = log.scrollHeight;
    }

    async function ask(text) {
      if (!text.trim()) return;
      setText(bubble('user'), text); input.value = ''; send.disabled = true;
      const t = bubble('bot typing'); t.textContent = 'Checking our help docs…';
      try {
        const res = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                               body: JSON.stringify({ session_id: session, message: text }) });
        t.remove();
        if (!res.ok) {
          const e = await res.json().catch(() => ({}));
          botMessage({ answer: typeof e.detail === 'string' ? e.detail : 'Something went wrong. Please try again.', action: 'unavailable', citations: [] });
          return;
        }
        botMessage(await res.json());
      } catch { t.remove(); botMessage({ answer: 'Connection problem. Please try again.', action: 'unavailable', citations: [] }); }
      finally { send.disabled = false; input.focus(); }
    }

    form.addEventListener('submit', e => { e.preventDefault(); ask(input.value); });
    if (suggest) suggest.querySelectorAll('button').forEach(b => b.addEventListener('click', () => ask(b.textContent)));
    botMessage({ answer: "Hi! I'm the Pinecrest support assistant. Ask me about shipping, returns, sizing, rentals or your order.", action: 'answer', citations: [] });
    return { ask };
  }

  window.SupportChat = { mount };
})();
