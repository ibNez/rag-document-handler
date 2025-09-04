// ============================================================================
// Consolidated JavaScript extracted from index.html
// ----------------------------------------------------------------------------
// Responsibilities:
//   - Periodic polling & UI update for: staging file processing, processed docs,
//     URL ingestion statuses, email ingestion statuses, deletion progress, stats panels
//   - User actions: delete file, process file, edit / delete email accounts
//   - Time localization for any element marked with .dt-local (shared with search.html)
// ----------------------------------------------------------------------------
// Template ↔ Function Cross‑Reference (index.html unless noted):
//   #staging-area .................. refreshStagingArea(), refreshProcessingStatus(), handleDelete(), handleProcess(), pollStagingDeletion()
//   .file-card (staging) ........... refreshProcessingStatus(), updateCardStatus(), handleDelete(), handleProcess(), pollStagingDeletion(), removeCard()
//   #processed-documents-section ... refreshProcessedDocuments()
//   #url-management ................ refreshUrlManagement(), refreshUrlStatuses(), finalizeUrlStatus(), updateProgressBar(), fadeRemoveRow(), handleDelete()
//   #email-accounts ................ refreshEmailAccounts(), attachEmailAccountListeners(), fillEditForm(), populateEditFormFromDataset(), refreshEmailStatuses(), finalizeEmailStatus(), updateProgressBar(), fadeRemoveRow()
//   #knowledgebase-stats ........... refreshKnowledgebaseStats()
//   Uploaded docs table (tr[data-uploaded-row]) ... pollUploadedDeletion(), handleDelete(), fadeRemoveRow()
//   Deletion progress (both tables + staging cards) ... pollUploadedDeletion(), pollStagingDeletion(), completionTracker Map
//   Global click actions (buttons) .. globalClickHandler() ⇒ handleDelete()/handleProcess()
//   Time stamps (.dt-local) ......... applyLocalTimes(), formatLocal(), normalizeUtc() (also used by search.html)
// ----------------------------------------------------------------------------
// Implementation Notes:
//   - All polling intervals are created inside the DOMContentLoaded block for isolation.
//   - For partial section refreshes we refetch the full page (cheap server-side render) and
//     swap only the innerHTML of the section container; this keeps server template source
//     of truth and avoids duplicating markup in JS.
//   - Defensive try/catch kept minimal to avoid swallowing critical errors; silent failures
//     are deliberate for polling endpoints that may return transient 404/not_found states.
//   - Functions are grouped below by concern with banner comments for quick scanning.
// ============================================================================

(function(){
  // Debug logging helper (enable/disable via window.rdhLog.enable()/disable() or localStorage key)
  let LOG_ENABLED = (localStorage.getItem('rdhLogEnabled') || '1') === '1';
  function log(msg){ if(LOG_ENABLED) console.log('[RDH]', new Date().toISOString(), msg); }
  window.rdhLog = {
    enable(){ LOG_ENABLED = true; localStorage.setItem('rdhLogEnabled','1'); log('Logging enabled'); },
    disable(){ log('Logging disabled'); LOG_ENABLED = false; localStorage.setItem('rdhLogEnabled','0'); },
    status(){ return LOG_ENABLED; }
  };
  log('index.js loaded');
  const completionTracker = new Map();
  const userLang = (navigator.language || 'en-US');
  const use12h = /^en-US/i.test(userLang);
  const dtf = new Intl.DateTimeFormat(userLang, { year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:use12h });

  // === Time Localization Utilities (shared with search.html) ===
  // normalizeUtc: Accepts a UTC-ish string (optionally missing 'Z'), returns Date or null.
  function normalizeUtc(utcString){
    let s = String(utcString || '').trim();
    if(!s) return null;
    if(!/[zZ]|[+\-]\d{2}:?\d{2}$/.test(s)) { s = s.replace(' ', 'T') + 'Z'; }
    const d = new Date(s); return isNaN(d.getTime()) ? null : d;
  }
  // formatLocal: Formats a UTC timestamp into the user's locale or fallback.
  function formatLocal(utcString){ const d = normalizeUtc(utcString); if(!d) return '—'; try { return dtf.format(d);} catch{ return d.toLocaleString(); } }
  // applyLocalTimes: Rewrites text content of all .dt-local elements within optional scope.
  function applyLocalTimes(scope=document){ scope.querySelectorAll('.dt-local[data-utc]').forEach(el => { el.textContent = formatLocal(el.getAttribute('data-utc')); }); }

  // === Processed Documents Panel ===
  // refreshProcessedDocuments: Replaces #processed-documents-section after a staging file finishes.
  function refreshProcessedDocuments(){
    log('refreshProcessedDocuments() start');
    fetch(window.location.href, { headers:{'Accept':'text/html'} })
      .then(r=>r.text())
      .then(html => {
        const doc = new DOMParser().parseFromString(html,'text/html');
        const newSec = doc.querySelector('#processed-documents-section');
        const cur = document.querySelector('#processed-documents-section');
        if(newSec && cur){ cur.innerHTML = newSec.innerHTML; applyLocalTimes(cur); log('Processed documents section updated'); }
      }).catch(e=> log('refreshProcessedDocuments() error: '+ e));
  }

  // === Staging File Processing & Status Cards ===
  // refreshProcessingStatus: Polls backend /status/<filename> for each staging card still in an active state.
  function refreshProcessingStatus(){
    document.querySelectorAll('.file-card').forEach(card => {
      const cardTitle = card.querySelector('.card-title');
      let filename='';
      if(cardTitle){ const clone=cardTitle.cloneNode(true); clone.querySelectorAll('.status-indicator').forEach(el=>el.remove()); filename=clone.textContent.trim(); }
      if(!filename) return;
      if(!card.querySelector('.status-processing, .status-chunking, .status-embedding, .status-storing')) return;
      fetch(`/status/${encodeURIComponent(filename)}`)
        .then(r=>r.json())
        .then(data => { if(data.status==='not_found') return; updateCardStatus(card, data, filename); })
        .catch(()=>{});
    });
  }

  // updateCardStatus: Updates progress, message, and indicator dot for a staging card (or processed card in edge cases).
  function updateCardStatus(card, data, filename){
    const indicator = card.querySelector('.status-indicator');
    if(indicator) indicator.className = `status-indicator status-${data.status}`;
    const stagingStatusArea = card.querySelector('.staging-status-area');
    const serverStatusArea = card.querySelector('.server-status-area');
    const actionButton = card.querySelector('.action-button');
    const isStaging = !!stagingStatusArea;

    // Hide/show action button based on processing status
    if(actionButton) {
      const isProcessing = ['processing', 'queued', 'chunking', 'embedding', 'storing'].includes(data.status);
      actionButton.style.display = isProcessing ? 'none' : 'inline-block';
    }

    if(isStaging){ 
      stagingStatusArea.style.display='block'; 
      // Hide server status when dynamic status is active
      if(serverStatusArea) serverStatusArea.style.display='none';
      
      let html='';
      if(data.message){ const cls = data.status==='error'? 'text-danger': data.status==='completed'? 'text-success':'text-info'; html += `<small class="d-block ${cls}">${data.message}</small>`; }
      if(typeof data.progress==='number' && data.progress>=0){ const pCls = data.status==='error'? 'bg-danger':'bg-success'; const anim = ['queued','processing','chunking','embedding','storing'].includes(data.status)?'progress-animated':''; html += `<div class="progress progress-custom mt-1" style="height:6px;"><div class="progress-bar ${pCls} ${anim}" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${data.progress}" style="width:${data.progress}%"></div></div>`; }
      stagingStatusArea.innerHTML = html; 
    }
    else { const messageEl = card.querySelector('small'); if(messageEl && data.message) messageEl.textContent = data.message; const pb=card.querySelector('.progress-bar'); if(pb && typeof data.progress==='number'){ pb.style.width=`${data.progress}%`; pb.setAttribute('aria-valuenow', String(data.progress)); if(['queued','processing','chunking','embedding','storing'].includes(data.status)) pb.classList.add('progress-animated'); else pb.classList.remove('progress-animated'); } }

    if(data.message && data.message.toLowerCase().includes('deletion complete')) { removeCard(card, isStaging); return; }
    if(['completed','error'].includes(data.status) && isStaging){ setTimeout(()=>{ removeCard(card, true); if(data.status==='completed') refreshProcessedDocuments(); },3000); }
  }
  // removeCard: Fades out and removes a staging or processed card wrapper column.
  function removeCard(card, isStaging){ const cardCol = isStaging? card.closest('.col-md-6'): card; if(!cardCol) return; cardCol.style.transition='opacity 200ms ease'; cardCol.style.opacity='0'; setTimeout(()=> cardCol.remove(),220); }

  // === URL Ingestion Status Table (#url-management) ===
  // refreshUrlStatuses: Polls each progress bar row for live ingestion or deletion progress.
  function refreshUrlStatuses(){ document.querySelectorAll('.progress-bar[data-url-id]').forEach(pb => { const id=pb.getAttribute('data-url-id'); fetch(`/url_status/${id}`).then(r=>r.json()).then(data => { if(!data) return; if(data.status==='deleted'){ fadeRemoveRow(pb); return;} if(data.status==='not_found'){ finalizeUrlStatus(pb, data); return;} const pct= typeof data.progress==='number'? data.progress:0; updateProgressBar(pb, pct, data.status); }).catch(()=>{}); }); }
  // finalizeUrlStatus: Replaces progress bar cell with a final badge after completion or not_found.
  function finalizeUrlStatus(pb,data){ try { const td=pb.closest('td'); if(td){ const statusText=(data.last_update_status||'—').replaceAll('_',' '); const cls = data.last_update_status==='updated'? 'bg-success':(['unchanged','no_content'].includes(data.last_update_status)? 'bg-secondary':'bg-light text-dark'); td.innerHTML=`<span class="badge ${cls}">${statusText.charAt(0).toUpperCase()+statusText.slice(1)}</span>`; } } catch(e){} }

  // === Email Ingestion Status Table (#email-accounts) ===
  // refreshEmailStatuses: Polls each email account ingestion row progress bar.
  function refreshEmailStatuses(){ document.querySelectorAll('.progress-bar[data-email-id]').forEach(pb => { const id=pb.getAttribute('data-email-id'); fetch(`/email_status/${id}`).then(r=>r.json()).then(data => { if(!data) return; if(data.status==='deleted'){ fadeRemoveRow(pb); return;} if(data.status==='not_found'){ finalizeEmailStatus(pb, data); return;} const pct= typeof data.progress==='number'? data.progress:0; updateProgressBar(pb, pct, data.status); }).catch(()=>{}); }); }
  // finalizeEmailStatus: Replaces progress bar cell with a final badge for email ingestion.
  function finalizeEmailStatus(pb,data){ try { const td=pb.closest('td'); if(td){ const statusText=data.last_update_status || '—'; const cls = statusText.startsWith('error')? 'bg-light text-dark':'bg-success'; td.innerHTML=`<span class="badge ${cls}">${statusText}</span>`; } } catch(e){} }

  // updateProgressBar: Shared utility for URL + Email ingestion progress rows.
  function updateProgressBar(pb, pct, status){ pb.style.width=`${pct}%`; pb.setAttribute('aria-valuenow', String(pct)); if(['queued','processing','chunking','embedding','storing'].includes(status)) pb.classList.add('progress-animated'); else pb.classList.remove('progress-animated'); if(['completed','error'].includes(status)){ const td=pb.closest('td'); if(td){ const s=status==='completed'? 'updated':'error'; const cls = s==='updated'? 'bg-success':'bg-light text-dark'; td.innerHTML=`<span class="badge ${cls}">${s.charAt(0).toUpperCase()+s.slice(1)}</span>`; } } }
  // fadeRemoveRow: Generic fade out helper for table rows.
  function fadeRemoveRow(pb){ const row = pb.closest('tr'); if(row){ row.style.transition='opacity 200ms ease'; row.style.opacity='0'; setTimeout(()=> row.remove(),220); } }

  // === Email Accounts Section (modal + list) ===
  // refreshEmailAccounts: Re-fetches #email-accounts block, then re-binds modal button listeners.
  function refreshEmailAccounts(){ fetch(window.location.href).then(r=>r.text()).then(html => { const doc=new DOMParser().parseFromString(html,'text/html'); const newSec=doc.querySelector('#email-accounts'); const cur=document.querySelector('#email-accounts'); if(newSec && cur){ cur.innerHTML=newSec.innerHTML; applyLocalTimes(cur); attachEmailAccountListeners(); } }); }
  // attachEmailAccountListeners: Binds click handlers for Edit/Delete account buttons to populate modals.
  function attachEmailAccountListeners(){ document.querySelectorAll('.edit-account-btn').forEach(btn => { btn.addEventListener('click', () => { fetch('/email_accounts').then(r=>r.json()).then(accounts => { const id=btn.dataset.id; const account=accounts.find(a=>a.id==id); if(account){ fillEditForm(account, btn.dataset.action); } else { populateEditFormFromDataset(btn); } }).catch(()=> populateEditFormFromDataset(btn)); }); }); document.querySelectorAll('.delete-account-btn').forEach(btn => { btn.addEventListener('click', () => { const form=document.getElementById('deleteEmailAccountForm'); form.action=btn.dataset.action; document.getElementById('deleteAccountName').textContent=btn.dataset.name||''; }); }); }
  // fillEditForm: Populates the edit modal with live account data from /email_accounts.
  function fillEditForm(account, action){ const form=document.getElementById('editEmailAccountForm'); form.action=action; const map={ editAccountId:account.id, editAccountName:account.account_name, editServer:account.server, editEmailAddress:account.email_address, editPassword:'', editPort:account.port, editMailbox:account.mailbox, editBatchLimit:account.batch_limit, editRefreshInterval:account.refresh_interval_minutes, editUseSSL:account.use_ssl, editServerType:account.server_type||'imap', editLastSyncedOffset:account.last_synced_offset||'0'}; Object.keys(map).forEach(id=>{ const el=document.getElementById(id); if(!el) return; if(el.type==='checkbox') el.checked=!!map[id]; else el.value = map[id] ?? ''; }); }
  // populateEditFormFromDataset: Fallback when live fetch fails; uses data-* attributes on the clicked button.
  function populateEditFormFromDataset(btn){ const form=document.getElementById('editEmailAccountForm'); form.action=btn.dataset.action; const fieldMap={ editAccountId:'id', editAccountName:'name', editServer:'server', editEmailAddress:'email', editPassword:null, editPort:'port', editMailbox:'mailbox', editBatchLimit:'batch', editRefreshInterval:'interval', editUseSSL:'ssl', editServerType:'type', editLastSyncedOffset:'offset'}; Object.entries(fieldMap).forEach(([id,key])=>{ const el=document.getElementById(id); if(!el) return; if(id==='editUseSSL') el.checked=['1','true','on'].includes(btn.dataset[key]); else if(id==='editPassword') el.value=''; else el.value= btn.dataset[key] || (id==='editServerType'?'imap':''); }); }

  // === Stats & Section Partial Refreshers ===
  // refreshKnowledgebaseStats: Swaps metrics cards inside #knowledgebase-stats.
  function refreshKnowledgebaseStats(){ log('refreshKnowledgebaseStats() start'); fetch(window.location.href).then(r=>r.text()).then(html => { const doc=new DOMParser().parseFromString(html,'text/html'); const newSec=doc.querySelector('#knowledgebase-stats'); const cur=document.querySelector('#knowledgebase-stats'); if(newSec && cur){ cur.innerHTML=newSec.innerHTML; applyLocalTimes(cur); log('knowledgebase stats updated'); } }).catch(e=> log('refreshKnowledgebaseStats() error: '+e)); }
  // refreshStagingArea: Replaces only staging files area (#staging-files-area) preserving upload form state.
  function refreshStagingArea(){ log('refreshStagingArea() start'); fetch(window.location.href).then(r=>r.text()).then(html => { const doc=new DOMParser().parseFromString(html,'text/html'); const newSec=doc.querySelector('#staging-files-area'); const cur=document.querySelector('#staging-files-area'); if(newSec && cur){ cur.innerHTML=newSec.innerHTML; applyLocalTimes(cur); log('staging area updated'); } }).catch(e=> log('refreshStagingArea() error: '+e)); }
  // refreshUrlManagement: Reloads URL management table while preserving the current input value.
  function refreshUrlManagement(){ 
    log('refreshUrlManagement() start'); 
    fetch(window.location.href)
      .then(r=>r.text())
      .then(html => { 
        const doc = new DOMParser().parseFromString(html, 'text/html'); 
        const newSec = doc.querySelector('#url-management'); 
        const cur = document.querySelector('#url-management'); 
        
        if(newSec && cur) {
          // Preserve URL input value and focus state
          const urlInput = cur.querySelector('input[name="url"]');
          const inputValue = urlInput?.value || '';
          const hadFocus = urlInput === document.activeElement;
          const cursorPosition = hadFocus ? urlInput.selectionStart : 0;
          
          // Update the content
          cur.innerHTML = newSec.innerHTML; 
          
          // Restore input state
          if(inputValue) {
            const newInput = cur.querySelector('input[name="url"]'); 
            if(newInput) {
              newInput.value = inputValue;
              // Restore focus and cursor position if the input had focus
              if(hadFocus) {
                newInput.focus();
                newInput.setSelectionRange(cursorPosition, cursorPosition);
              }
            }
          }
          
          applyLocalTimes(cur); 
          log('url management section updated, input preserved'); 
        } 
      })
      .catch(e=> log('refreshUrlManagement() error: '+e)); 
  }

  // === Deletion Polling (Uploaded + Staging) ===
  // pollUploadedDeletion: Tracks long-running embedding cleanups for already-uploaded documents table rows.
  function pollUploadedDeletion(){ document.querySelectorAll('tr[data-uploaded-row]').forEach(row => { const filename=row.getAttribute('data-uploaded-row'); const actionCell=row.lastElementChild; if(!actionCell.querySelector('.fa-spinner')) return; fetch(`/status/${encodeURIComponent(filename)}`).then(r=>r.json()).then(data => { if(!data) return; const statusCell=row.children[row.children.length-2]; let html=''; if(data.message){ html += `<div class="small mb-1">${data.message}`; if(data.deletion_status && data.deletion_status.remaining_records>0){ html += ` (${data.deletion_status.remaining_records} embeddings remaining)`; } html+='</div>'; } if(typeof data.progress==='number' && data.progress>0){ html += `<div class=\"progress progress-custom\" style=\"height:6px;\"><div class=\"progress-bar bg-danger progress-animated\" role=\"progressbar\" aria-valuemin=0 aria-valuemax=100 aria-valuenow=${data.progress} style=\"width:${data.progress}%\"></div></div>`; } statusCell.innerHTML=html; const completionMessage = data.status==='completed' && (data.message||'').toLowerCase().includes('deletion complete - all embeddings cleaned up'); const cleanupComplete = data.deletion_status && data.deletion_status.cleanup_complete === true; const both=completionMessage && cleanupComplete; const key=filename; const now=Date.now(); if(both){ if(!completionTracker.has(key)){ completionTracker.set(key, now); return; } const elapsed=now-completionTracker.get(key); if(elapsed>=10000){ fadeRemoveRow(row.querySelector('.progress-bar')||{closest:()=>row}); completionTracker.delete(key); } } else { completionTracker.delete(key); } }).catch(()=>{}); }); }
  // pollStagingDeletion: Same concept as pollUploadedDeletion but for staging cards (before processing completion).
  function pollStagingDeletion(){ document.querySelectorAll('.staging-status-area[data-filename]').forEach(area => { if(area.style.display==='none') return; const filename=area.getAttribute('data-filename'); fetch(`/status/${encodeURIComponent(filename)}`).then(r=>r.json()).then(data => { if(!data) return; let html=''; if(data.message){ html += `<small class='d-block'>${data.message}`; if(data.deletion_status && data.deletion_status.remaining_records>0){ html += ` (${data.deletion_status.remaining_records} embeddings remaining)`; } html+='</small>'; } if(typeof data.progress==='number' && data.progress>0){ html += `<div class=\"progress progress-custom mt-1\" style=\"height:6px;\"><div class=\"progress-bar bg-danger progress-animated\" role=\"progressbar\" aria-valuemin=0 aria-valuemax=100 aria-valuenow=${data.progress} style=\"width:${data.progress}%\"></div></div>`; } area.innerHTML=html; const completionMessage = data.status==='completed' && (data.message||'').toLowerCase().includes('deletion complete'); const cleanupComplete = data.deletion_status && data.deletion_status.cleanup_complete === true; const both=completionMessage && cleanupComplete; const key=`staging_${filename}`; const now=Date.now(); if(both){ if(!completionTracker.has(key)){ completionTracker.set(key, now); return; } const elapsed=now-completionTracker.get(key); if(elapsed>=10000){ const card=area.closest('.file-card'); if(card){ const col=card.closest('.col-md-6'); if(col){ col.style.transition='opacity 200ms ease'; col.style.opacity='0'; setTimeout(()=>col.remove(),220); } } completionTracker.delete(key); } } else { completionTracker.delete(key); } }).catch(()=>{}); }); }

  // === Global UI Event Handling ===
  // globalClickHandler: Event delegation for delete/process buttons; minimizes individual listeners.
  function globalClickHandler(e){ const del = e.target.closest('.delete-file-btn'); if(del){ handleDelete(del); return; } const proc = e.target.closest('.process-file-btn'); if(proc){ handleProcess(proc); return; } }
  // handleDelete: Orchestrates UI changes and POST request for file deletion (uploaded or staging).
  function handleDelete(btn){ 
    const filename=btn.getAttribute('data-filename'); 
    const url=btn.getAttribute('data-url'); 
    const isUploaded=url.includes('/uploaded/'); 
    const isStaging=url.includes('/staging/'); 
    const msg = isUploaded ? 'Delete this file and remove from database?\n\nNote: Milvus will clean up embeddings asynchronously in the background.' : 'Delete this file?\n\nNote: Any embeddings will be cleaned up asynchronously by Milvus.'; 
    if(!confirm(msg)) return; 
    
    if(isUploaded){ 
      const row = btn.closest('tr'); 
      const cells=row.querySelectorAll('td'); 
      const statusCell=cells[cells.length-2]; 
      const actionCell=cells[cells.length-1]; 
      actionCell.innerHTML='<button type="button" class="btn btn-outline-danger btn-sm" disabled><i class="fas fa-spinner fa-spin"></i></button>'; 
      statusCell.innerHTML='<div class="small text-muted">Deletion command sent - Milvus cleanup is async</div>'; 
    } else if(isStaging){ 
      const card=btn.closest('.file-card'); 
      const dropdown=card.querySelector('.dropdown'); 
      const statusArea=card.querySelector('.staging-status-area'); 
      const serverStatusArea = card.querySelector('.server-status-area');
      
      dropdown.innerHTML='<button class="btn btn-sm btn-outline-secondary" disabled><i class="fas fa-spinner fa-spin"></i></button>'; 
      statusArea.style.display='block'; 
      // Hide server status when deletion starts
      if(serverStatusArea) serverStatusArea.style.display='none';
      statusArea.innerHTML='<small class="d-block text-muted">Deletion command sent</small>'; 
    }
    
    fetch(url,{ method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'} })
      .then(r=>{ if(r.ok){ setTimeout(()=>{ if(isUploaded){ const row=document.querySelector(`tr[data-uploaded-row="${filename}"]`); if(row){ row.style.transition='opacity 200ms ease'; row.style.opacity='0'; setTimeout(()=>row.remove(),220);} } else if(isStaging){ const card=document.querySelector(`.staging-status-area[data-filename="${filename}"]`)?.closest('.file-card'); if(card){ const col=card.closest('.col-md-6'); if(col){ col.style.transition='opacity 200ms ease'; col.style.opacity='0'; setTimeout(()=>col.remove(),220);} } } },1000);} else { alert('Delete request failed.'); location.reload(); } })
      .catch(()=>{ alert('Delete request failed.'); location.reload(); }); 
  }
  // handleProcess: Sends a processing request for a staging file and updates its card status indicator.
  function handleProcess(btn){ 
    const filename=btn.getAttribute('data-filename'); 
    const url=btn.getAttribute('data-url'); 
    const card=btn.closest('.file-card'); 
    const dropdown=card.querySelector('.dropdown'); 
    const actionButton = card.querySelector('.action-button');
    const statusArea=card.querySelector('.staging-status-area'); 
    const serverStatusArea = card.querySelector('.server-status-area');
    
    // Hide action button immediately when processing starts
    if(actionButton) actionButton.style.display = 'none';
    
    dropdown.innerHTML='<button class="btn btn-sm btn-outline-secondary" disabled><i class="fas fa-spinner fa-spin"></i></button>'; 
    statusArea.style.display='block'; 
    // Hide server status when processing starts
    if(serverStatusArea) serverStatusArea.style.display='none';
    statusArea.innerHTML='<small class="d-block text-muted">Starting processing...</small>'; 
    
    fetch(url,{ headers:{'Accept':'application/json'} }).then(r=>{ 
      if(r.ok){ 
        statusArea.innerHTML='<small class="d-block text-success">File queued for processing</small>'; 
        const title=card.querySelector('.card-title'); 
        if(title && !card.querySelector('.status-indicator')){ 
          title.innerHTML += ' <span class="status-indicator status-processing">●</span>'; 
        } 
      } else { 
        alert('Process request failed.'); 
        location.reload(); 
      } 
    }).catch(()=>{ 
      alert('Process request failed.'); 
      location.reload(); 
    }); 
  }

  // === Bootstrap: attach listeners & schedule pollers ===
  document.addEventListener('DOMContentLoaded', () => {
    applyLocalTimes();
    attachEmailAccountListeners();
    document.addEventListener('click', globalClickHandler);
    setInterval(refreshProcessingStatus, 2000);
    setInterval(refreshUrlStatuses, 1500);
    setInterval(refreshEmailStatuses, 1500);
    setInterval(pollUploadedDeletion, 5000);
    setInterval(pollStagingDeletion, 5000);
  setInterval(()=>{ log('Interval tick: refreshEmailAccounts'); refreshEmailAccounts(); }, 10000);
  setInterval(()=>{ log('Interval tick: refreshKnowledgebaseStats'); refreshKnowledgebaseStats(); }, 10000);
  setInterval(()=>{ log('Interval tick: refreshStagingArea'); refreshStagingArea(); }, 10000);
  setInterval(()=>{ log('Interval tick: refreshUrlManagement'); refreshUrlManagement(); }, 10000);
  setInterval(()=>{ log('Interval tick: refreshProcessedDocuments'); refreshProcessedDocuments(); }, 10000);
  });
})();
