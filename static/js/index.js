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
      if(!card.querySelector('.status-processing, .status-chunking, .status-embedding, .status-storing, .status-connecting, .status-preparing, .status-syncing, .status-finalizing')) return;
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
      const isProcessing = ['processing', 'queued', 'chunking', 'embedding', 'storing', 'connecting', 'preparing', 'syncing', 'finalizing'].includes(data.status);
      actionButton.style.display = isProcessing ? 'none' : 'inline-block';
    }

    if(isStaging){ 
      stagingStatusArea.style.display='block'; 
      // Hide server status when dynamic status is active
      if(serverStatusArea) serverStatusArea.style.display='none';
      
      let html='';
      if(data.message){ const cls = data.status==='error'? 'text-danger': data.status==='completed'? 'text-success':'text-info'; html += `<small class="d-block ${cls}">${data.message}</small>`; }
      if(typeof data.progress==='number' && data.progress>=0){ const pCls = data.status==='error'? 'bg-danger':'bg-success'; const anim = ['queued','processing','chunking','embedding','storing','connecting','preparing','syncing','finalizing'].includes(data.status)?'progress-animated':''; html += `<div class="progress progress-custom mt-1" style="height:6px;"><div class="progress-bar ${pCls} ${anim}" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${data.progress}" style="width:${data.progress}%"></div></div>`; }
      stagingStatusArea.innerHTML = html; 
    }
    else { const messageEl = card.querySelector('small'); if(messageEl && data.message) messageEl.textContent = data.message; const pb=card.querySelector('.progress-bar'); if(pb && typeof data.progress==='number'){ pb.style.width=`${data.progress}%`; pb.setAttribute('aria-valuenow', String(data.progress)); if(['queued','processing','chunking','embedding','storing','connecting','preparing','syncing','finalizing'].includes(data.status)) pb.classList.add('progress-animated'); else pb.classList.remove('progress-animated'); } }

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
  function updateProgressBar(pb, pct, status){ pb.style.width=`${pct}%`; pb.setAttribute('aria-valuenow', String(pct)); if(['queued','processing','chunking','embedding','storing','connecting','preparing','syncing','finalizing'].includes(status)) pb.classList.add('progress-animated'); else pb.classList.remove('progress-animated'); if(['completed','error'].includes(status)){ const td=pb.closest('td'); if(td){ const s=status==='completed'? 'updated':'error'; const cls = s==='updated'? 'bg-success':'bg-light text-dark'; td.innerHTML=`<span class="badge ${cls}">${s.charAt(0).toUpperCase()+s.slice(1)}</span>`; } } }
  // fadeRemoveRow: Generic fade out helper for table rows.
  function fadeRemoveRow(pb){ const row = pb.closest('tr'); if(row){ row.style.transition='opacity 200ms ease'; row.style.opacity='0'; setTimeout(()=> row.remove(),220); } }

  // === Email Accounts Section (modal + list) ===
  // refreshEmailAccounts: Re-fetches #email-accounts block, then re-binds modal button listeners.
  function refreshEmailAccounts(){ fetch(window.location.href).then(r=>r.text()).then(html => { const doc=new DOMParser().parseFromString(html,'text/html'); const newSec=doc.querySelector('#email-accounts'); const cur=document.querySelector('#email-accounts'); if(newSec && cur){ cur.innerHTML=newSec.innerHTML; applyLocalTimes(cur); attachEmailAccountListeners(); } }); }
  // attachEmailAccountListeners: Binds click handlers for Edit/Delete account buttons to populate modals.
  function attachEmailAccountListeners(){ 
    document.querySelectorAll('.edit-account-btn').forEach(btn => { 
      btn.addEventListener('click', () => { 
        console.log('🔍 Edit button clicked, fetching account data...');
        fetch('/email_accounts')
          .then(r => {
            console.log('🔍 /email_accounts response status:', r.status);
            return r.json();
          })
          .then(accounts => { 
            console.log('🔍 Received accounts:', accounts);
            console.log('🔍 First account structure:', accounts[0]);
            const id = btn.dataset.id; 
            const account = accounts.find(a => a.id == id || a.email_account_id == id); 
            console.log('🔍 Looking for account ID:', id);
            console.log('🔍 Found account:', account);
            if(account){ 
              fillEditForm(account, btn.dataset.action); 
            } else { 
              console.log('🔍 Account not found, falling back to dataset');
              populateEditFormFromDataset(btn); 
            } 
          })
          .catch((error) => {
            console.log('🔍 /email_accounts fetch failed, falling back to dataset:', error);
            populateEditFormFromDataset(btn);
          }); 
      }); 
    }); 
    document.querySelectorAll('.delete-account-btn').forEach(btn => { 
      btn.addEventListener('click', () => { 
        const form = document.getElementById('deleteEmailAccountForm'); 
        form.action = btn.dataset.action; 
        document.getElementById('deleteAccountName').textContent = btn.dataset.name || ''; 
      }); 
    }); 
  }
  // fillEditForm: Populates the edit modal with live account data from /email_accounts.
  function fillEditForm(account, action) {
    const form = document.getElementById('editEmailAccountForm');
    form.action = action;
    
    const serverType = account.server_type || 'imap';
    
    // Show/hide field sets based on server type
    const traditionalFields = document.getElementById('editTraditionalFields');
    const gmailFields = document.getElementById('editGmailFields');
    
    if (serverType === 'gmail') {
      // Show Gmail fields, hide traditional fields
      if (traditionalFields) traditionalFields.classList.add('d-none');
      if (gmailFields) gmailFields.classList.remove('d-none');
      
      // Remove required attribute from hidden traditional fields
      const traditionalRequiredFields = ['editServer', 'editEmailAddress', 'editPort', 'editMailbox'];
      traditionalRequiredFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) field.removeAttribute('required');
      });
      
      // Ensure Gmail fields have required attribute
      const gmailRequiredFields = ['editAccountName', 'editGmailEmailAddress', 'editGmailMailbox'];
      gmailRequiredFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) field.setAttribute('required', 'required');
      });
      
      // Populate Gmail-specific fields
      const gmailFieldMap = {
        editAccountName: account.account_name,
        editGmailEmailAddress: account.email_address,
        editGmailMailbox: account.mailbox,
        editBatchSize: account.batch_size,
        editRefreshInterval: account.refresh_interval_minutes,
        editAccountId: account.id
      };
      
      Object.keys(gmailFieldMap).forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = gmailFieldMap[id] ?? '';
      });
      
      // Try to load Gmail labels for selection
      if (account.id) {
        console.log('🔍 About to call loadGmailLabelsForEdit with account.mailbox:', account.mailbox);
        loadGmailLabelsForEdit(account.id, account.mailbox);
      }
      
    } else {
      // Show traditional fields, hide Gmail fields
      if (traditionalFields) traditionalFields.classList.remove('d-none');
      if (gmailFields) gmailFields.classList.add('d-none');
      
      // Populate traditional fields
      const traditionalFieldMap = {
        editAccountId: account.id,
        editAccountName: account.account_name,
        editServer: account.server,
        editEmailAddress: account.email_address,
        editPassword: '',
        editPort: account.port,
        editMailbox: account.mailbox,
        editBatchSize: account.batch_size,
        editRefreshInterval: account.refresh_interval_minutes,
        editUseSSL: account.use_ssl,
        editLastSyncedOffset: account.last_synced_offset || '0'
      };
      
      Object.keys(traditionalFieldMap).forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        if (el.type === 'checkbox') el.checked = !!traditionalFieldMap[id];
        else el.value = traditionalFieldMap[id] ?? '';
      });
      
      // Handle mailbox selection for IMAP/Exchange
      handleEditServerTypeChange(serverType, account);
    }
  }
  // populateEditFormFromDataset: Fallback when live fetch fails; uses data-* attributes on the clicked button.
  function populateEditFormFromDataset(btn) {
    const form = document.getElementById('editEmailAccountForm');
    form.action = btn.dataset.action;
    
    const serverType = btn.dataset.type || 'imap';
    
    // Show/hide field sets based on server type
    const traditionalFields = document.getElementById('editTraditionalFields');
    const gmailFields = document.getElementById('editGmailFields');
    
    if (serverType === 'gmail') {
      // Show Gmail fields, hide traditional fields
      if (traditionalFields) traditionalFields.classList.add('d-none');
      if (gmailFields) gmailFields.classList.remove('d-none');
      
      // Remove required attribute from hidden traditional fields
      const traditionalRequiredFields = ['editServer', 'editEmailAddress', 'editPort', 'editMailbox'];
      traditionalRequiredFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) field.removeAttribute('required');
      });
      
      // Ensure Gmail fields have required attribute
      const gmailRequiredFields = ['editAccountName', 'editGmailEmailAddress', 'editGmailMailbox'];
      gmailRequiredFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) field.setAttribute('required', 'required');
      });
      
      // Populate Gmail-specific fields
      const gmailFieldMap = {
        editAccountName: 'name',
        editGmailEmailAddress: 'email',
        editGmailMailbox: 'mailbox',
        editBatchSize: 'batch',
        editRefreshInterval: 'interval',
        editAccountId: 'id'
      };
      
      Object.entries(gmailFieldMap).forEach(([id, key]) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.value = btn.dataset[key] || '';
      });
      
      // Try to load Gmail labels for selection
      const accountId = btn.dataset.id;
      const currentMailbox = btn.dataset.mailbox;
      console.log('🔍 Gmail account detected in populateEditFormFromDataset, calling loadGmailLabelsForEdit');
      console.log('🔍 Account data from button dataset:', { accountId, currentMailbox });
      console.log('🔍 All button dataset:', btn.dataset);
      if (accountId) {
        loadGmailLabelsForEdit(accountId, currentMailbox);
      } else {
        console.error('❌ No account ID found in dataset for Gmail label loading');
      }
      
    } else {
      // Show traditional fields, hide Gmail fields
      if (traditionalFields) traditionalFields.classList.remove('d-none');
      if (gmailFields) gmailFields.classList.add('d-none');
      
      // Restore required attribute to traditional fields
      const traditionalRequiredFields = ['editServer', 'editEmailAddress', 'editPort', 'editMailbox'];
      traditionalRequiredFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) field.setAttribute('required', 'required');
      });
      
      // Remove required from hidden Gmail fields
      const gmailRequiredFields = ['editGmailEmailAddress', 'editGmailMailbox'];
      gmailRequiredFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) field.removeAttribute('required');
      });
      
      // Populate traditional fields
      const fieldMap = {
        editAccountId: 'id',
        editAccountName: 'name',
        editServer: 'server',
        editEmailAddress: 'email',
        editPassword: null,
        editPort: 'port',
        editMailbox: 'mailbox',
        editBatchSize: 'batch',
        editRefreshInterval: 'interval',
        editUseSSL: 'ssl',
        editLastSyncedOffset: 'offset'
      };
      
      Object.entries(fieldMap).forEach(([id, key]) => {
        const el = document.getElementById(id);
        if (!el) return;
        if (id === 'editUseSSL') el.checked = ['1', 'true', 'on'].includes(btn.dataset[key]);
        else if (id === 'editPassword') el.value = '';
        else el.value = btn.dataset[key] || '';
      });
      
      // Handle mailbox selection for IMAP/Exchange
      const account = {
        id: btn.dataset.id,
        mailbox: btn.dataset.mailbox,
        server_type: serverType
      };
      handleEditServerTypeChange(serverType, account);
    }
  }

  // handleEditServerTypeChange: Handle server type changes in edit modal
  function handleEditServerTypeChange(serverType, account) {
    const mailboxField = document.getElementById('editMailbox');
    const dynamicMailboxesDiv = document.getElementById('editDynamicMailboxes');
    
    if (!mailboxField || !dynamicMailboxesDiv) return;
    
    // Only show dynamic mailbox selection for IMAP and Exchange (which work)
    // Gmail needs OAuth token implementation first
    if ((serverType === 'imap' || serverType === 'exchange') && account && account.id) {
      // Hide traditional mailbox field, show dynamic selection
      mailboxField.style.display = 'none';
      dynamicMailboxesDiv.classList.remove('d-none');
      
      // Load mailboxes for this account
      loadEditAccountMailboxes(account.id, account.mailbox, serverType);
    } else {
      // Show traditional mailbox field for Gmail and other types
      mailboxField.style.display = 'block';
      dynamicMailboxesDiv.classList.add('d-none');
      
      // Show helpful placeholder for Gmail
      if (serverType === 'gmail') {
        mailboxField.placeholder = 'e.g., INBOX,SENT,Important (comma-separated Gmail labels)';
      } else {
        mailboxField.placeholder = 'e.g., INBOX';
      }
    }
  }

  // Gmail helper function for updating mailbox field with selected labels
  function updateGmailMailboxField() {
    console.log('🔍 updateGmailMailboxField called');
    const checkboxes = document.querySelectorAll('input[name="gmail_labels"]:checked');
    const selectedLabels = Array.from(checkboxes).map(cb => cb.value);
    const mailboxField = document.getElementById('editGmailMailbox');
    
    console.log('🔍 Selected labels:', selectedLabels);
    console.log('🔍 Mailbox field found:', !!mailboxField);
    
    if (mailboxField) {
      mailboxField.value = selectedLabels.join(', ');
      console.log('🔍 Updated mailbox field value to:', mailboxField.value);
    }
  }

  // Gmail label functions for edit modal
  function loadGmailLabelsForEdit(accountId, currentMailbox) {
    console.log('🔍 loadGmailLabelsForEdit called with:', accountId, currentMailbox);
    const dynamicLabels = document.getElementById('editGmailDynamicLabels');
    const loadingIndicator = document.getElementById('editGmailLabelsLoading');
    const labelsList = document.getElementById('editGmailLabelsList');
    const errorDiv = document.getElementById('editGmailLabelsError');
    
    console.log('🔍 Elements found:', {
      dynamicLabels: !!dynamicLabels,
      loadingIndicator: !!loadingIndicator,
      labelsList: !!labelsList,
      errorDiv: !!errorDiv
    });
    
    if (!dynamicLabels) {
      console.error('❌ editGmailDynamicLabels element not found!');
      return;
    }
    
    // Show loading state
    console.log('🔍 Showing loading state');
    if (loadingIndicator) loadingIndicator.classList.remove('d-none');
    if (labelsList) labelsList.classList.add('d-none');
    if (errorDiv) errorDiv.classList.add('d-none');
    
    console.log('🔍 Making fetch request to:', `/email_accounts/${accountId}/mailboxes`);
    fetch(`/email_accounts/${accountId}/mailboxes`)
      .then(response => {
        console.log('🔍 Response status:', response.status);
        console.log('🔍 Response headers:', response.headers);
        return response.json();
      })
      .then(data => {
        console.log('🔍 Response data:', data);
        if (data.error) {
          throw new Error(data.error);
        }
        
        // Extract mailboxes array from response
        const mailboxes = data.mailboxes || data;
        if (!Array.isArray(mailboxes)) {
          throw new Error('Invalid response format: expected mailboxes array');
        }
        
        // Build checkboxes for each label
        const currentLabels = currentMailbox ? currentMailbox.split(',').map(l => l.trim()) : [];
        console.log('🔍 Processing currentMailbox:', currentMailbox);
        console.log('🔍 Parsed currentLabels:', currentLabels);
        
        const checkboxesHtml = mailboxes.map(label => {
          const isChecked = currentLabels.includes(label.name) ? 'checked' : '';
          console.log(`🔍 Label ${label.name}: currentLabels.includes = ${currentLabels.includes(label.name)}, isChecked = "${isChecked}"`);
          return `
            <div class="form-check">
              <input class="form-check-input" type="checkbox" 
                     value="${label.name}" 
                     id="gmail_label_${label.id}" 
                     name="gmail_labels" 
                     ${isChecked}>
              <label class="form-check-label" for="gmail_label_${label.id}">
                ${label.name} <small class="text-muted">(${label.messageCount} messages)</small>
              </label>
            </div>
          `;
        }).join('');
        
        if (labelsList) {
          labelsList.innerHTML = checkboxesHtml;
          labelsList.classList.remove('d-none');
        }
        
        // Update the mailbox field with selected values on change
        const checkboxes = labelsList.querySelectorAll('input[name="gmail_labels"]');
        console.log('🔍 Attaching event listeners to', checkboxes.length, 'Gmail label checkboxes');
        checkboxes.forEach((checkbox, index) => {
          console.log('🔍 Attaching listener to checkbox', index, 'label:', checkbox.value);
          checkbox.addEventListener('change', () => {
            console.log('🔍 Checkbox changed:', checkbox.value, 'checked:', checkbox.checked);
            updateGmailMailboxField();
          });
        });
        
        // Call updateGmailMailboxField once initially to set the field value
        updateGmailMailboxField();
        
      })
      .catch(error => {
        console.error('❌ Gmail labels loading failed:', error);
        console.error('❌ Error details:', error.message, error.stack);
        // Don't hide the error - let it bubble up
        if (loadingIndicator) loadingIndicator.classList.add('d-none');
        throw error; // Re-throw to expose the problem
      })
      .finally(() => {
        if (loadingIndicator) loadingIndicator.classList.add('d-none');
      });
  }

  // loadEditAccountMailboxes: Load mailboxes for edit modal (works for all server types)
  function loadEditAccountMailboxes(accountId, currentMailboxes, serverType) {
    const loadingDiv = document.getElementById('editMailboxesLoading');
    const listDiv = document.getElementById('editMailboxesList');
    const errorDiv = document.getElementById('editMailboxesError');
    
    if (!loadingDiv || !listDiv || !errorDiv) return;
    
    // Show loading, hide others
    loadingDiv.classList.remove('d-none');
    listDiv.classList.add('d-none');
    errorDiv.classList.add('d-none');
    
    fetch(`/email_accounts/${accountId}/mailboxes`)
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          throw new Error(data.error);
        }
        
        // Hide loading, show list
        loadingDiv.classList.add('d-none');
        listDiv.classList.remove('d-none');
        
        // Generate checkboxes for mailboxes
        const currentSelection = currentMailboxes ? currentMailboxes.split(',') : ['INBOX'];
        let html = '';
        
        data.mailboxes.forEach(mailbox => {
          const isChecked = currentSelection.includes(mailbox.id) ? 'checked' : '';
          const messageCount = mailbox.messageCount ? ` (${mailbox.messageCount})` : '';
          const typeIcon = getMailboxTypeIcon(mailbox.type);
          
          html += `
            <div class="form-check">
              <input class="form-check-input" type="checkbox" value="${mailbox.id}" 
                     id="editMailbox_${mailbox.id}" name="mailboxes[]" ${isChecked}>
              <label class="form-check-label" for="editMailbox_${mailbox.id}">
                ${typeIcon} ${mailbox.name}${messageCount}
              </label>
            </div>
          `;
        });
        
        if (html === '') {
          html = '<div class="text-muted">No mailboxes found</div>';
        }
        
        listDiv.innerHTML = html;
        
        // Update hidden mailbox field when checkboxes change
        const checkboxes = listDiv.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => {
          cb.addEventListener('change', updateEditMailboxField);
        });
        
        // Initial update
        updateEditMailboxField();
        
      })
      .catch(error => {
        console.error('Failed to load mailboxes:', error);
        
        // Hide loading, show error  
        loadingDiv.classList.add('d-none');
        
        // Check if this is a Gmail fallback case
        if (error.message.includes('OAuth implementation') || error.message.includes('fallback')) {
          errorDiv.classList.remove('d-none');
          errorDiv.innerHTML = `
            <strong>Gmail Label Discovery Not Available</strong><br>
            Gmail requires OAuth token storage for automatic label discovery. 
            Please manually enter the mailbox/labels you want to sync (e.g., "INBOX,SENT,Important").
          `;
        } else {
          errorDiv.classList.remove('d-none');
          errorDiv.textContent = `Failed to load mailboxes: ${error.message}`;
        }
        
        // Fall back to showing the traditional mailbox field
        const mailboxField = document.getElementById('editMailbox');
        if (mailboxField) {
          mailboxField.style.display = 'block';
        }
        const dynamicMailboxesDiv = document.getElementById('editDynamicMailboxes');
        if (dynamicMailboxesDiv) {
          dynamicMailboxesDiv.classList.add('d-none');
        }
      });
  }

  // getMailboxTypeIcon: Get icon for mailbox type
  function getMailboxTypeIcon(type) {
    const icons = {
      'inbox': '📥',
      'sent': '📤', 
      'drafts': '📝',
      'trash': '🗑️',
      'spam': '🚫',
      'outbox': '📮',
      'folder': '📁',
      'label': '🏷️'
    };
    return icons[type] || '📁';
  }

  // updateEditMailboxField: Update the hidden mailbox field with selected mailboxes
  function updateEditMailboxField() {
    const listDiv = document.getElementById('editMailboxesList');
    const mailboxField = document.getElementById('editMailbox');
    
    if (!listDiv || !mailboxField) return;
    
    const selectedMailboxes = [];
    const checkboxes = listDiv.querySelectorAll('input[type="checkbox"]:checked');
    checkboxes.forEach(cb => {
      selectedMailboxes.push(cb.value);
    });
    
    // Update the hidden field with comma-separated values
    mailboxField.value = selectedMailboxes.length > 0 ? selectedMailboxes.join(',') : 'INBOX';
  }

  // === Email Account Form Handlers ===
  // setupEmailAccountFormHandlers: Sets up AJAX form submission handlers for add/edit forms
  function setupEmailAccountFormHandlers() {
    // Handle Add Email Account form
    const addForm = document.getElementById('addEmailAccountForm');
    if (addForm) {
      addForm.addEventListener('submit', function(e) {
        e.preventDefault();
        // Check if button is disabled to prevent multiple submissions
        const submitBtn = document.getElementById('addEmailAccountSubmit');
        if (submitBtn && submitBtn.disabled) {
          return false;
        }
        handleEmailAccountSubmission(addForm, 'addEmailAccountError', 'addEmailAccountSuccess', 'addEmailAccountSubmit');
      });
    }

    // Handle Edit Email Account form
    const editForm = document.getElementById('editEmailAccountForm');
    if (editForm) {
      editForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Ensure Gmail mailbox field is updated before submission
        const gmailFieldsVisible = !document.getElementById('editGmailFields')?.classList.contains('d-none');
        if (gmailFieldsVisible) {
          console.log('🔍 Gmail fields visible, updating mailbox field before submission');
          updateGmailMailboxField();
        }
        
        // Debug: Check Gmail mailbox field value before submission
        const gmailMailboxField = document.getElementById('editGmailMailbox');
        if (gmailMailboxField) {
          console.log('🔍 Gmail mailbox field value before submission:', gmailMailboxField.value);
          console.log('🔍 Gmail mailbox field name:', gmailMailboxField.name);
        }
        
        // Check if button is disabled to prevent multiple submissions
        const submitBtn = document.getElementById('editEmailAccountSubmit');
        if (submitBtn && submitBtn.disabled) {
          return false;
        }
        handleEmailAccountSubmission(editForm, 'editEmailAccountError', 'editEmailAccountSuccess', 'editEmailAccountSubmit');
      });
    }

    // Clear messages when modals are opened
    const addModal = document.getElementById('addEmailAccountModal');
    const editModal = document.getElementById('editEmailAccountModal');
    
    if (addModal) {
      addModal.addEventListener('show.bs.modal', function() {
        const errorDiv = document.getElementById('addEmailAccountError');
        const successDiv = document.getElementById('addEmailAccountSuccess');
        const submitBtn = document.getElementById('addEmailAccountSubmit');
        
        if (errorDiv) errorDiv.classList.add('d-none');
        if (successDiv) successDiv.classList.add('d-none');
        
        // Reset button state when modal opens
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.style.pointerEvents = 'auto';
          submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span> Save';
        }
      });
    }
    
    if (editModal) {
      editModal.addEventListener('show.bs.modal', function() {
        const errorDiv = document.getElementById('editEmailAccountError');
        const successDiv = document.getElementById('editEmailAccountSuccess');
        const submitBtn = document.getElementById('editEmailAccountSubmit');
        
        if (errorDiv) errorDiv.classList.add('d-none');
        if (successDiv) successDiv.classList.add('d-none');
        
        // Reset button state when modal opens
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.style.pointerEvents = 'auto';
          submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span> Save changes';
        }
      });
    }
  }

  // === Dynamic Edit Email Form Behavior ===
  // setupDynamicEditEmailForm: Sets up dynamic mailbox selection for edit modal
  function setupDynamicEditEmailForm() {
    // Handle initial state when modal is shown
    const editModal = document.getElementById('editEmailAccountModal');
    if (editModal) {
      editModal.addEventListener('shown.bs.modal', function() {
        // The server type and mailbox handling is now managed in fillEditForm
        // and populateEditFormFromDataset functions, so no additional setup needed here
      });
    }
  }

  // === Dynamic Email Form Behavior ===
  // setupDynamicEmailForm: Sets up server type dependent form fields
  function setupDynamicEmailForm() {
    const serverTypeSelect = document.getElementById('addServerType');
    const traditionalFields = document.getElementById('addTraditionalFields');
    const gmailFields = document.getElementById('addGmailFields');
    const commonFields = document.getElementById('addCommonFields');
    const gmailAuthBtn = document.getElementById('addGmailAuthBtn');
    const submitBtn = document.getElementById('addEmailAccountSubmit');

    if (!serverTypeSelect) return;

    // Initial state: hide save button until server type is selected
    if (submitBtn) submitBtn.style.display = 'none';

    serverTypeSelect.addEventListener('change', function() {
      const serverType = this.value;
      
      // Hide all conditional fields
      if (traditionalFields) traditionalFields.classList.add('d-none');
      if (gmailFields) gmailFields.classList.add('d-none');
      if (commonFields) commonFields.classList.add('d-none');
      if (gmailAuthBtn) gmailAuthBtn.style.display = 'none';
      if (submitBtn) submitBtn.style.display = 'none'; // Hide until appropriate server type selected

      // Clear required attributes
      document.querySelectorAll('#addTraditionalFields [required], #addGmailFields [required]').forEach(el => {
        el.removeAttribute('required');
      });

      // Show appropriate fields based on server type
      if (serverType === 'gmail') {
        if (gmailFields) gmailFields.classList.remove('d-none');
        if (commonFields) commonFields.classList.remove('d-none');
        if (gmailAuthBtn) gmailAuthBtn.style.display = 'inline-block';
        if (submitBtn) submitBtn.style.display = 'none'; // Hide Save button for Gmail - OAuth handles saving

        // Set required for Gmail fields
        const gmailAddress = document.querySelector('#addGmailFields input[name="gmail_address"]');
        if (gmailAddress) gmailAddress.setAttribute('required', 'required');

      } else if (serverType === 'imap' || serverType === 'exchange') {
        if (traditionalFields) traditionalFields.classList.remove('d-none');
        if (commonFields) commonFields.classList.remove('d-none');
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.style.display = 'inline-block'; // Show Save button for traditional auth
        }

        // Set required for traditional fields
        document.querySelectorAll('#addTraditionalFields input[name="server"], #addTraditionalFields input[name="email_address"], #addTraditionalFields input[name="password"], #addTraditionalFields input[name="port"]').forEach(el => {
          el.setAttribute('required', 'required');
        });

        // Pre-fill common values for known providers
        if (serverType === 'imap') {
          const serverField = document.querySelector('#addTraditionalFields input[name="server"]');
          const portField = document.querySelector('#addTraditionalFields input[name="port"]');
          const sslField = document.querySelector('#addTraditionalFields input[name="use_ssl"]');
          
          if (serverField && !serverField.value) serverField.placeholder = 'e.g., imap.gmail.com';
          if (portField && !portField.value) portField.value = '993';
          if (sslField) sslField.checked = true;
        }
      }
    });

    // Gmail OAuth button handler
    if (gmailAuthBtn) {
      gmailAuthBtn.addEventListener('click', function() {
        const gmailAddress = document.querySelector('#addGmailFields input[name="gmail_address"]').value;
        if (!gmailAddress) {
          alert('Please enter your Gmail address first.');
          return;
        }
        
        // Start OAuth flow
        initiateGmailOAuth(gmailAddress);
      });
    }
  }

  // initiateGmailOAuth: Starts the Gmail OAuth flow
  function initiateGmailOAuth(emailAddress) {
    const gmailAuthBtn = document.getElementById('addGmailAuthBtn');
    const authStatus = document.getElementById('gmailAuthStatus');
    
    // Clear any previous status messages
    if (authStatus) {
      authStatus.classList.add('d-none');
    }
    
    // Show loading state
    gmailAuthBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Opening Gmail...';
    gmailAuthBtn.disabled = true;

    // Build OAuth URL with email hint
    const oauthUrl = `/oauth/gmail/start?email=${encodeURIComponent(emailAddress)}`;
    
    // Open OAuth in popup window
    const popup = window.open(oauthUrl, 'gmail_oauth', 'width=500,height=600,scrollbars=yes,resizable=yes');
    
    if (!popup) {
      alert('Popup blocked! Please allow popups for this site and try again.');
      // Reset button state
      gmailAuthBtn.disabled = false;
      gmailAuthBtn.innerHTML = '<i class="fab fa-google"></i> Authenticate with Gmail';
      return;
    }
    
    // Listen for OAuth completion
    const checkClosed = setInterval(() => {
      if (popup.closed) {
        clearInterval(checkClosed);
        
        // Check if OAuth was successful by checking for success message
        checkGmailOAuthStatus();
        
        // Reset button state if OAuth failed (success will hide the button)
        const oauthSuccess = localStorage.getItem('gmail_oauth_success');
        if (!oauthSuccess) {
          gmailAuthBtn.disabled = false;
          gmailAuthBtn.innerHTML = '<i class="fab fa-google"></i> Authenticate with Gmail';
        }
      }
    }, 1000);
  }

  // checkGmailOAuthStatus: Checks if Gmail OAuth was successful
  function checkGmailOAuthStatus() {
    // Check localStorage for OAuth success (set by OAuth callback)
    const oauthSuccess = localStorage.getItem('gmail_oauth_success');
    if (oauthSuccess) {
      const gmailEmail = localStorage.getItem('gmail_oauth_email');
      localStorage.removeItem('gmail_oauth_success');
      localStorage.removeItem('gmail_oauth_email');
      
      // Show success message
      const authStatus = document.getElementById('gmailAuthStatus');
      const authBtn = document.getElementById('addGmailAuthBtn');
      
      if (authStatus) {
        authStatus.classList.remove('d-none');
        authStatus.innerHTML = `
          <div class="alert alert-success">
            <i class="fas fa-check-circle"></i>
            Gmail authentication successful! Saving account...
          </div>
        `;
      }
      
      // Hide auth button since OAuth is complete
      if (authBtn) authBtn.style.display = 'none';
      
      // Populate the Gmail address field if we got the email
      if (gmailEmail) {
        const gmailAddressField = document.getElementById('gmail_address');
        if (gmailAddressField) {
          gmailAddressField.value = gmailEmail;
        }
      }
      
      // Auto-submit the form for Gmail accounts
      setTimeout(() => {
        const form = document.getElementById('addEmailAccountForm');
        if (form) {
          handleEmailAccountSubmission(form, 'addEmailAccountError', 'addEmailAccountSuccess', 'addEmailAccountSubmit');
        }
      }, 1000); // Brief delay to show the success message
    }
  }

  // handleEmailAccountSubmission: Common handler for add/edit email account forms
  function handleEmailAccountSubmission(form, errorId, successId, submitId) {
    const errorDiv = document.getElementById(errorId);
    const successDiv = document.getElementById(successId);
    const submitBtn = document.getElementById(submitId);
    const spinner = submitBtn.querySelector('.spinner-border');

    // Prevent multiple submissions if already processing
    if (submitBtn.disabled) {
      return;
    }

    // Hide previous messages
    errorDiv.classList.add('d-none');
    successDiv.classList.add('d-none');

    // Show loading state and disable button to prevent double clicks
    submitBtn.disabled = true;
    submitBtn.style.pointerEvents = 'none'; // Extra protection against clicks
    spinner.classList.remove('d-none');
    
    // Update button text to show validation in progress
    const originalText = submitBtn.textContent.trim();
    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Validating...';

    // Prepare form data
    const formData = new FormData(form);
    
    // Debug: Log all form data before submission
    console.log('🔍 Form data entries before submission:');
    for (let [key, value] of formData.entries()) {
      console.log(`  ${key}: ${value}`);
    }

    // Submit via AJAX
    fetch(form.action, {
      method: 'POST',
      headers: {
        'X-Requested-With': 'XMLHttpRequest'
      },
      body: formData
    })
    .then(response => {
      console.log('🔍 Server response status:', response.status);
      console.log('🔍 Server response headers:', response.headers);
      return response.json();
    })
    .then(data => {
      console.log('🔍 Server response data:', data);
      if (data.success) {
        // Hide error message before showing success
        errorDiv.classList.add('d-none');
        
        successDiv.textContent = data.message;
        successDiv.classList.remove('d-none');
        
        // Update button to show success state
        submitBtn.innerHTML = '<i class="fas fa-check"></i> Success!';
        
        // Reset form and close modal after a brief delay
        setTimeout(() => {
          form.reset();
          const modal = form.closest('.modal');
          if (modal) {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            if (modalInstance) {
              modalInstance.hide();
            }
          }
          // Refresh the email accounts section
          refreshEmailAccounts();
        }, 1500);
      } else {
        // Hide success message before showing error
        successDiv.classList.add('d-none');
        
        errorDiv.textContent = data.error || 'An error occurred';
        errorDiv.classList.remove('d-none');
        
        // Re-enable button for retry after validation failure
        resetButtonState();
      }
    })
    .catch(error => {
      console.error('🔍 Network error occurred:', error);
      console.error('🔍 Error details:', error.message, error.stack);
      
      // Hide success message before showing error
      successDiv.classList.add('d-none');
      
      errorDiv.textContent = 'Network error: ' + error.message;
      errorDiv.classList.remove('d-none');
      
      // Re-enable button for retry after network error
      resetButtonState();
    });
    
    // Helper function to reset button state for retry
    function resetButtonState() {
      submitBtn.disabled = false;
      submitBtn.style.pointerEvents = 'auto';
      spinner.classList.add('d-none');
      submitBtn.innerHTML = originalText;
    }
  }

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

  // resetEmailAccountButtonStates: Reset button states for all email account forms on page load
  function resetEmailAccountButtonStates() {
    const buttons = ['addEmailAccountSubmit', 'editEmailAccountSubmit'];
    buttons.forEach(buttonId => {
      const button = document.getElementById(buttonId);
      if (button) {
        button.disabled = false;
        button.style.pointerEvents = 'auto';
        
        // Reset to original text content based on button ID
        if (buttonId === 'addEmailAccountSubmit') {
          button.innerHTML = '<span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span> Save';
        } else if (buttonId === 'editEmailAccountSubmit') {
          button.innerHTML = '<span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true"></span> Save changes';
        }
      }
    });
  }

  // === Bootstrap: attach listeners & schedule pollers ===
  document.addEventListener('DOMContentLoaded', () => {
    applyLocalTimes();
    attachEmailAccountListeners();
    setupEmailAccountFormHandlers();
    setupDynamicEmailForm();
    setupDynamicEditEmailForm();
    
    // Reset any email account form button states on page load
    resetEmailAccountButtonStates();
    
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
