(() => {
  const seedImageInput = document.getElementById('seedImageInput');
  const selectSeedBtn = document.getElementById('selectSeedBtn');
  const seedFileName = document.getElementById('seedFileName');
  const referenceStrip = document.getElementById('referenceStrip');
  const parentPostInput = document.getElementById('parentPostInput');
  const applyParentBtn = document.getElementById('applyParentBtn');
  const previewShell = document.getElementById('previewShell');
  const currentGallery = document.getElementById('currentGallery');
  const previewEmpty = document.getElementById('previewEmpty');
  const currentParentId = document.getElementById('currentParentId');
  const currentMode = document.getElementById('currentMode');
  const editPromptInput = document.getElementById('editPromptInput');
  const submitEditBtn = document.getElementById('submitEditBtn');
  const resetCycleBtn = document.getElementById('resetCycleBtn');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  const editStatusText = document.getElementById('editStatusText');
  const historyCount = document.getElementById('historyCount');
  const historyEmpty = document.getElementById('historyEmpty');
  const historyList = document.getElementById('historyList');
  const editProgressWrap = document.getElementById('editProgressWrap');
  const editProgressBar = document.getElementById('editProgressBar');
  const editProgressText = document.getElementById('editProgressText');
  const REFERENCE_LIMIT = 3;

  const state = {
    editing: false,
    referenceImages: [],
    currentImageUrl: '',
    currentImageUrls: [],
    currentParentPostId: '',
    currentSourceImageUrl: '',
    currentModeValue: 'upload',
    history: [],
    editRound: 0,
  };
  let workbenchEditAbortController = null;
  let editProgressTimer = null;
  let editProgressHideTimer = null;
  let editProgressValue = 0;
  let dragCounter = 0;

  function toast(message, type) {
    if (typeof showToast === 'function') {
      showToast(message, type);
    }
  }

  function getParentMemoryApi() {
    return window.ParentPostMemory || null;
  }

  function rememberParentPost(entry) {
    const api = getParentMemoryApi();
    if (!api || !entry) return;
    try {
      api.remember(entry);
    } catch (e) {
      // ignore
    }
  }

  async function copyText(value) {
    const text = String(value || '').trim();
    if (!text) return false;
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', 'readonly');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    textarea.style.pointerEvents = 'none';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const copied = document.execCommand('copy');
    document.body.removeChild(textarea);
    return copied;
  }

  function shortId(value) {
    const raw = String(value || '').trim();
    if (!raw) return '-';
    if (raw.length <= 14) return raw;
    return `${raw.slice(0, 7)}...${raw.slice(-7)}`;
  }

  function formatTime(ts) {
    try {
      return new Date(ts).toLocaleString('zh-CN', {
        hour12: false,
      });
    } catch (e) {
      return '-';
    }
  }

  function maskPrompt(prompt) {
    const raw = String(prompt || '').trim();
    if (!raw) return '-';
    return raw;
  }

  function buildImaginePublicUrl(parentPostId) {
    return `https://imagine-public.x.ai/imagine-public/images/${parentPostId}.jpg`;
  }

  async function fetchParentPostInfo(authHeader, parentPostId) {
    const pid = String(parentPostId || '').trim();
    if (!pid) return null;
    const response = await fetch(`/v1/public/imagine/parent-post?parent_post_id=${encodeURIComponent(pid)}`, {
      headers: {
        Authorization: authHeader,
      },
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error((data && (data.detail || data.message)) || `查询 parentPostId 失败 (${response.status})`);
    }
    return data;
  }

  function normalizeHttpSourceUrl(value) {
    const raw = String(value || '').trim();
    if (!raw) return '';
    if (raw.startsWith('data:')) return '';
    if (raw.startsWith('http://') || raw.startsWith('https://')) {
      return raw;
    }
    if (raw.startsWith('/')) {
      return `${window.location.origin}${raw}`;
    }
    return '';
  }

  function pickSourceImageUrl(hit, parentPostId, fallbackValue = '') {
    const candidates = [
      hit && hit.current_source_image_url,
      hit && hit.currentSourceImageUrl,
      hit && hit.source_image_url,
      hit && hit.sourceImageUrl,
      hit && hit.image_url,
      hit && hit.imageUrl,
      fallbackValue,
    ];
    for (const candidate of candidates) {
      const normalized = normalizeHttpSourceUrl(candidate);
      if (normalized) return normalized;
    }
    return parentPostId ? buildImaginePublicUrl(parentPostId) : '';
  }

  function pickPreviewUrl(hit, parentPostId) {
    const candidates = [
      hit && hit.imageUrl,
      hit && hit.image_url,
      hit && hit.url,
      hit && hit.sourceImageUrl,
      hit && hit.source_image_url,
    ];
    for (const candidate of candidates) {
      const raw = String(candidate || '').trim();
      if (raw) return raw;
    }
    const source = pickSourceImageUrl(hit, parentPostId);
    return source || (parentPostId ? buildImaginePublicUrl(parentPostId) : '');
  }

  function extractParentPostId(value) {
    const raw = String(value || '').trim();
    if (!raw) return '';
    const direct = raw.match(/^[0-9a-fA-F-]{32,36}$/);
    if (direct) return direct[0];
    const generated = raw.match(/\/generated\/([0-9a-fA-F-]{32,36})(?:\/|$)/);
    if (generated) return generated[1];
    const imaginePublic = raw.match(/\/imagine-public\/images\/([0-9a-fA-F-]{32,36})(?:\.jpg|\/|$)/);
    if (imaginePublic) return imaginePublic[1];
    const imagePath = raw.match(/\/images\/([0-9a-fA-F-]{32,36})(?:\.jpg|\/|$)/);
    if (imagePath) return imagePath[1];
    const all = raw.match(/([0-9a-fA-F-]{32,36})/g);
    return all && all.length ? all[all.length - 1] : '';
  }

  function resolveParentMemoryByText(text) {
    const input = String(text || '').trim();
    if (!input) return null;
    const api = getParentMemoryApi();
    if (api && typeof api.resolveByText === 'function') {
      try {
        const hit = api.resolveByText(input);
        if (hit && hit.parentPostId) {
          const parentPostId = String(hit.parentPostId || '').trim();
          return {
            ...hit,
            parentPostId,
          };
        }
        return hit;
      } catch (e) {
        // ignore
      }
    }
    const parentPostId = extractParentPostId(input);
    if (!parentPostId) return null;
    return {
      parentPostId,
      sourceImageUrl: buildImaginePublicUrl(parentPostId),
      imageUrl: buildImaginePublicUrl(parentPostId),
      origin: 'fallback',
    };
  }

  async function applyParentPostFromText(text, options = {}) {
    const silent = Boolean(options.silent);
    const addToReferences = Boolean(options.addToReferences);
    let hit = resolveParentMemoryByText(text);
    if (!hit || !hit.parentPostId) {
      if (!silent) {
        toast('未识别到有效 parentPostId', 'warning');
      }
      return false;
    }
    const parentPostId = String(hit.parentPostId || '').trim();
    let authHeader = '';
    try {
      authHeader = await ensurePublicAuth();
    } catch (e) {
      if (!silent) {
        toast(String(e.message || e), 'error');
      }
      return false;
    }
    try {
      const remoteHit = await fetchParentPostInfo(authHeader, parentPostId);
      if (remoteHit && remoteHit.parent_post_id) {
        hit = {
          ...hit,
          parentPostId,
          current_source_image_url: remoteHit.source_image_url,
          source_image_url: remoteHit.source_image_url,
          sourceImageUrl: remoteHit.source_image_url,
          imageUrl: remoteHit.thumbnail_image_url || remoteHit.media_url || hit.imageUrl,
          image_url: remoteHit.thumbnail_image_url || remoteHit.media_url || hit.imageUrl,
        };
      }
    } catch (e) {
      if (!silent) {
        toast(String(e.message || e), 'warning');
      }
    }
    const sourceImageUrl = pickSourceImageUrl(hit, parentPostId);
    const previewUrl = pickPreviewUrl(hit, parentPostId);

    state.currentParentPostId = parentPostId;
    state.currentSourceImageUrl = sourceImageUrl;
    state.currentModeValue = 'parent_post';
    setPreview(previewUrl);
    updateMeta();
    setStatus('done', `已加载 parentPostId：${shortId(parentPostId)}`);
    if (parentPostInput) {
      parentPostInput.value = parentPostId;
    }
    rememberParentPost({
      parentPostId,
      sourceImageUrl,
      imageUrl: previewUrl,
      origin: 'workbench_paste_apply',
    });
    if (addToReferences) {
      const refUrl = String(sourceImageUrl || previewUrl || '').trim();
      if (!refUrl) {
        if (!silent) {
          toast('该 parentPostId 未解析到可用参考图', 'warning');
        }
      } else if (state.referenceImages.length >= REFERENCE_LIMIT) {
        if (!silent) {
          toast(`最多支持 ${REFERENCE_LIMIT} 张参考图`, 'warning');
        }
      } else {
        const exists = state.referenceImages.some((item) => (
          (item.parentPostId && item.parentPostId === parentPostId)
          || String(item.data || '').trim() === refUrl
        ));
        if (!exists) {
          state.referenceImages.push({
            id: buildRefId(),
            name: `parent:${shortId(parentPostId)}`,
            mime: '',
            data: refUrl,
            source: 'parent_post',
            parentPostId,
            sourceImageUrl: sourceImageUrl || refUrl,
            isPrimary: false,
            createdAt: Date.now(),
          });
          if (!state.referenceImages.some((item) => item.isPrimary)) {
            state.referenceImages[0].isPrimary = true;
          }
          normalizeReferenceOrder();
          renderReferenceStrip();
        }
      }
    }
    if (!silent) {
      toast(addToReferences ? '已载入 parentPostId，并加入参考图' : '已载入 parentPostId，可直接继续编辑', 'success');
    }
    return true;
  }

  function setStatus(stateName, text) {
    if (!editStatusText) return;
    editStatusText.textContent = text;
    editStatusText.classList.remove('running', 'done', 'error');
    if (stateName) {
      editStatusText.classList.add(stateName);
    }
  }

  function clearEditProgressTimer() {
    if (editProgressTimer) {
      clearInterval(editProgressTimer);
      editProgressTimer = null;
    }
    if (editProgressHideTimer) {
      clearTimeout(editProgressHideTimer);
      editProgressHideTimer = null;
    }
  }

  function setEditProgress(value, text) {
    const safe = Math.max(0, Math.min(100, Math.round(value || 0)));
    editProgressValue = safe;
    if (editProgressBar) {
      editProgressBar.style.width = `${safe}%`;
    }
    if (editProgressText) {
      editProgressText.textContent = text || `编辑中 ${safe}%`;
    }
  }

  function showEditProgress() {
    if (editProgressWrap) {
      editProgressWrap.classList.add('active');
      editProgressWrap.classList.remove('is-success', 'is-error');
    }
    if (editProgressText) {
      editProgressText.classList.add('active');
    }
    setEditProgress(4, '编辑中 4%');
  }

  function hideEditProgress() {
    clearEditProgressTimer();
    if (editProgressWrap) {
      editProgressWrap.classList.remove('active', 'is-success', 'is-error');
    }
    if (editProgressText) {
      editProgressText.classList.remove('active');
      editProgressText.textContent = '编辑中 0%';
    }
    if (editProgressBar) {
      editProgressBar.style.width = '0%';
    }
    editProgressValue = 0;
  }

  function startEditProgress() {
    clearEditProgressTimer();
    showEditProgress();
  }

  function finishEditProgress(success, text) {
    clearEditProgressTimer();
    if (!editProgressWrap) return;
    editProgressWrap.classList.add('active');
    editProgressWrap.classList.remove('is-success', 'is-error');
    editProgressWrap.classList.add(success ? 'is-success' : 'is-error');
    if (editProgressText) {
      editProgressText.classList.add('active');
    }
    setEditProgress(100, text || (success ? '编辑完成 100%' : '编辑失败'));
    editProgressHideTimer = setTimeout(() => {
      hideEditProgress();
      editProgressHideTimer = null;
    }, 900);
  }

  async function readFileAsDataUrl(file) {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('读取文件失败'));
      reader.readAsDataURL(file);
    });
  }

  function buildRefId() {
    return `ref_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
  }

  function getPrimaryReference() {
    if (!state.referenceImages.length) return null;
    return state.referenceImages[0];
  }

  function normalizeReferenceOrder() {
    if (!state.referenceImages.length) return;
    state.referenceImages.sort((a, b) => {
      return a.createdAt - b.createdAt;
    });
  }

  function updateReferenceSummary() {
    if (!seedFileName) return;
    seedFileName.textContent = `已添加 ${state.referenceImages.length}/${REFERENCE_LIMIT} 张`;
  }

  function renderReferenceStrip() {
    if (!referenceStrip) return;
    referenceStrip.innerHTML = '';
    if (!state.referenceImages.length) {
      const empty = document.createElement('div');
      empty.className = 'reference-empty';
      empty.textContent = '可上传 / 粘贴 / 拖拽参考图（最多 3 张）';
      referenceStrip.appendChild(empty);
      updateReferenceSummary();
      return;
    }

    state.referenceImages.forEach((item) => {
      const card = document.createElement('div');
      card.className = 'reference-item';
      card.title = `预览：${item.name}`;
      card.dataset.id = item.id;
      card.setAttribute('role', 'button');
      card.setAttribute('tabindex', '0');

      const img = document.createElement('img');
      img.className = 'reference-thumb';
      img.src = item.data;
      img.alt = item.name || 'reference';
      img.loading = 'lazy';
      card.appendChild(img);

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'reference-remove-btn';
      removeBtn.textContent = '×';
      removeBtn.title = '删除';
      removeBtn.dataset.id = item.id;
      removeBtn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        removeReferenceImage(item.id);
      });
      card.appendChild(removeBtn);

      card.addEventListener('click', () => {
        previewReference(item.id);
      });
      card.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          previewReference(item.id);
        }
      });
      referenceStrip.appendChild(card);
    });

    if (state.referenceImages.length < REFERENCE_LIMIT && seedImageInput) {
      const addSlot = document.createElement('button');
      addSlot.type = 'button';
      addSlot.className = 'reference-add-slot';
      addSlot.title = '继续添加';
      addSlot.textContent = '+';
      addSlot.addEventListener('click', () => seedImageInput.click());
      referenceStrip.appendChild(addSlot);
    }
    updateReferenceSummary();
  }

  function previewReference(id) {
    if (!id) return;
    const hit = state.referenceImages.find((item) => item.id === id);
    const previewData = String(hit && hit.data ? hit.data : '').trim();
    if (!previewData) return;

    // 仅存在这张 parentPostId 参考图时，点击即恢复到 parentPostId 编辑模式
    if (
      hit
      && hit.source === 'parent_post'
      && String(hit.parentPostId || '').trim()
      && state.referenceImages.length === 1
    ) {
      state.currentParentPostId = String(hit.parentPostId || '').trim();
      state.currentSourceImageUrl = String(hit.sourceImageUrl || previewData).trim();
      state.currentModeValue = 'parent_post';
      if (parentPostInput) {
        parentPostInput.value = state.currentParentPostId;
      }
      updateMeta();
      rememberParentPost({
        parentPostId: state.currentParentPostId,
        sourceImageUrl: state.currentSourceImageUrl,
        imageUrl: previewData,
        origin: 'workbench_reference_restore',
      });
      setPreview(previewData);
      setStatus('done', `已恢复 parentPostId 模式：${shortId(state.currentParentPostId)}`);
      return;
    }

    setPreview(previewData);
    setStatus('', '已预览参考图');
  }

  function setPrimaryReference(id) {
    // 兼容旧调用，点击行为仅预览，不再改变主图
    previewReference(id);
  }

  function removeReferenceImage(id) {
    const before = state.referenceImages.length;
    state.referenceImages = state.referenceImages.filter((item) => item.id !== id);
    if (state.referenceImages.length === before) return;
    normalizeReferenceOrder();
    renderReferenceStrip();
    if (!state.referenceImages.length) {
      setStatus('', '参考图已清空');
    }
  }

  function clearReferenceImages() {
    state.referenceImages = [];
    if (seedImageInput) seedImageInput.value = '';
    renderReferenceStrip();
  }

  function pickImageFilesFromDataTransfer(dataTransfer) {
    if (!dataTransfer) return [];
    const files = [];
    const pushIfImage = (file) => {
      if (!file) return;
      if (!String(file.type || '').startsWith('image/')) return;
      files.push(file);
    };
    if (dataTransfer.files && dataTransfer.files.length) {
      Array.from(dataTransfer.files).forEach(pushIfImage);
    }
    if (!files.length && dataTransfer.items && dataTransfer.items.length) {
      Array.from(dataTransfer.items).forEach((item) => {
        if (!item || item.kind !== 'file') return;
        const file = item.getAsFile ? item.getAsFile() : null;
        pushIfImage(file);
      });
    }
    return files;
  }

  async function addReferenceFiles(files, source) {
    if (!Array.isArray(files) || !files.length) return 0;
    if (state.editing) {
      toast('编辑进行中，暂时不能修改参考图', 'warning');
      return 0;
    }

    const slotsLeft = Math.max(0, REFERENCE_LIMIT - state.referenceImages.length);
    if (slotsLeft <= 0) {
      toast(`最多支持 ${REFERENCE_LIMIT} 张参考图`, 'warning');
      return 0;
    }
    const targets = files.slice(0, slotsLeft);
    const ignoredCount = Math.max(0, files.length - targets.length);

    let added = 0;
    for (const file of targets) {
      const mimeType = String(file.type || '');
      if (mimeType && !mimeType.startsWith('image/')) continue;
      const dataUrl = await readFileAsDataUrl(file);
      if (!dataUrl.startsWith('data:image/')) continue;
      state.referenceImages.push({
        id: buildRefId(),
        name: file.name || source || '未命名图片',
        mime: file.type || '',
        data: dataUrl,
        source: source || 'upload',
        isPrimary: false,
        createdAt: Date.now() + added,
      });
      added += 1;
    }
    normalizeReferenceOrder();
    renderReferenceStrip();

    if (added > 0) {
      clearHistory();
      resetCycle(false);
      setStatus('', '已载入参考图，可开始编辑');
    }

    if (ignoredCount > 0) {
      toast(`最多支持 ${REFERENCE_LIMIT} 张参考图，已忽略超出部分`, 'warning');
    }
    return added;
  }

  function setDragActive(active) {
    if (!previewShell) return;
    previewShell.classList.toggle('dragover', Boolean(active));
  }

  function hasFiles(dataTransfer) {
    if (!dataTransfer) return false;
    if (dataTransfer.files && dataTransfer.files.length > 0) return true;
    const types = dataTransfer.types;
    if (!types) return false;
    return Array.from(types).includes('Files');
  }

  function updateMeta() {
    if (currentParentId) {
      currentParentId.textContent = state.currentParentPostId || '-';
    }
    if (currentMode) {
      currentMode.textContent = state.currentModeValue || '-';
    }
    updateReferenceSummary();
  }

  function setPreviewImages(urls) {
    const list = (Array.isArray(urls) ? urls : [])
      .map((item) => String(item || '').trim())
      .filter(Boolean);
    state.currentImageUrls = list;
    state.currentImageUrl = list[0] || '';

    if (!currentGallery || !previewEmpty) return;
    currentGallery.innerHTML = '';
    const primaryUrl = list[0] || '';
    if (!primaryUrl) {
      currentGallery.dataset.count = '0';
      currentGallery.classList.add('hidden');
      previewEmpty.classList.remove('hidden');
      return;
    }

    currentGallery.dataset.count = '1';
    const item = document.createElement('div');
    item.className = 'current-gallery-item';
    const img = document.createElement('img');
    img.src = primaryUrl;
    img.alt = 'result-current';
    img.loading = 'lazy';
    img.decoding = 'async';
    item.appendChild(img);
    currentGallery.appendChild(item);

    currentGallery.classList.remove('hidden');
    previewEmpty.classList.add('hidden');
  }

  function setPreview(url) {
    const displayUrl = String(url || '').trim();
    if (!displayUrl) {
      setPreviewImages([]);
      return;
    }
    setPreviewImages([displayUrl]);
  }

  function setEditing(loading) {
    state.editing = Boolean(loading);
    if (submitEditBtn) {
      submitEditBtn.disabled = false;
      submitEditBtn.dataset.running = state.editing ? '1' : '0';
      submitEditBtn.textContent = state.editing ? '中止' : '执行编辑';
      submitEditBtn.classList.toggle('is-editing', state.editing);
      submitEditBtn.removeAttribute('aria-disabled');
    }
    if (selectSeedBtn) selectSeedBtn.disabled = state.editing;
    if (resetCycleBtn) resetCycleBtn.disabled = state.editing;
    if (clearHistoryBtn) clearHistoryBtn.disabled = state.editing;
    if (editPromptInput) editPromptInput.disabled = state.editing;

    if (state.editing) {
      setStatus('running', '编辑中...');
    }
  }

  function forceSubmitButtonAbortClickable() {
    if (!submitEditBtn) return;
    if (!state.editing) return;
    submitEditBtn.disabled = false;
    submitEditBtn.dataset.running = '1';
    submitEditBtn.textContent = '中止';
    submitEditBtn.classList.add('is-editing');
    submitEditBtn.removeAttribute('aria-disabled');
  }

  function setCurrentFromEntry(entry) {
    if (!entry) return;
    state.currentParentPostId = String(entry.parentPostId || '').trim();
    state.currentSourceImageUrl = String(entry.sourceImageUrl || '').trim();
    state.currentModeValue = String(entry.mode || '').trim() || 'parent_post';
    if (parentPostInput) {
      parentPostInput.value = state.currentParentPostId;
    }
    setPreview(entry.imageUrl || '');
    rememberParentPost({
      parentPostId: state.currentParentPostId,
      sourceImageUrl: state.currentSourceImageUrl,
      imageUrl: entry.imageUrl || '',
      origin: 'workbench_history_apply',
    });
    updateMeta();
    setStatus('done', `已切换到历史版本 #${entry.round}`);
  }

  function renderHistory() {
    if (!historyList || !historyEmpty || !historyCount) return;
    historyList.innerHTML = '';
    historyCount.textContent = `${state.history.length} 条`;

    if (!state.history.length) {
      historyEmpty.classList.remove('hidden');
      return;
    }
    historyEmpty.classList.add('hidden');

    state.history.forEach((entry) => {
      const item = document.createElement('div');
      item.className = 'history-item';

      const thumb = document.createElement('img');
      thumb.className = 'history-thumb';
      thumb.src = entry.imageUrl || '';
      thumb.alt = `history-${entry.round}`;
      thumb.loading = 'lazy';
      thumb.decoding = 'async';

      const main = document.createElement('div');
      main.className = 'history-main';

      const line1 = document.createElement('div');
      line1.className = 'history-line';
      const roundLabel = entry.roundTotal && entry.roundTotal > 1
        ? `#${entry.round}-${entry.roundIndex || 1}`
        : `#${entry.round}`;
      line1.innerHTML = `<strong>${roundLabel}</strong> 路 ${formatTime(entry.createdAt)} 路 ${entry.elapsedMs}ms`;

      const line2 = document.createElement('div');
      line2.className = 'history-line';
      line2.innerHTML = `mode=<strong>${entry.mode}</strong> 路 parentPostId=<strong>${shortId(entry.parentPostId)}</strong>`;

      const prompt = document.createElement('div');
      prompt.className = 'history-prompt';
      prompt.textContent = maskPrompt(entry.prompt);

      const actions = document.createElement('div');
      actions.className = 'history-actions';

      const applyBtn = document.createElement('button');
      applyBtn.type = 'button';
      applyBtn.className = 'geist-button-outline';
      applyBtn.textContent = '设为当前';
      applyBtn.addEventListener('click', () => {
        setCurrentFromEntry(entry);
      });

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.className = 'geist-button-outline';
      copyBtn.textContent = '复制ID';
      copyBtn.addEventListener('click', async () => {
        try {
          if (!entry.parentPostId) {
            toast('当前记录没有 parentPostId', 'warning');
            return;
          }
          const copied = await copyText(entry.parentPostId);
          if (!copied) {
            throw new Error('copy_failed');
          }
          toast('已复制 parentPostId', 'success');
        } catch (e) {
          toast('复制失败', 'error');
        }
      });

      actions.appendChild(applyBtn);
      actions.appendChild(copyBtn);

      main.appendChild(line1);
      main.appendChild(line2);
      main.appendChild(prompt);
      main.appendChild(actions);

      item.appendChild(thumb);
      item.appendChild(main);
      historyList.appendChild(item);
    });
  }

  function resetCycle(keepSeedPreview = true) {
    state.currentParentPostId = '';
    state.currentSourceImageUrl = '';
    state.currentModeValue = 'upload';
    if (parentPostInput) {
      parentPostInput.value = '';
    }
    updateMeta();

    const primary = getPrimaryReference();
    if (keepSeedPreview && primary && primary.data) {
      setPreview(primary.data);
    } else {
      setPreview('');
    }
    setStatus('', '未开始');
  }

  function clearHistory() {
    state.history = [];
    state.editRound = 0;
    renderHistory();
  }

  async function parseErrorText(res) {
    const text = await res.text();
    if (!text) return `请求失败：HTTP ${res.status}`;
    try {
      const data = JSON.parse(text);
      if (data && typeof data === 'object' && data.detail) {
        return String(data.detail);
      }
    } catch (e) {
      // ignore
    }
    return text;
  }

  async function requestWorkbenchEditStream(authHeader, body, onProgress, signal) {
    const payload = {
      ...body,
      stream: true,
    };
    const res = await fetch('/v1/public/imagine/workbench/edit', {
      method: 'POST',
      headers: {
        ...buildAuthHeaders(authHeader),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!res.ok) {
      throw new Error(await parseErrorText(res));
    }

    const contentType = String(res.headers.get('content-type') || '').toLowerCase();
    if (!contentType.includes('text/event-stream')) {
      return await res.json();
    }

    const reader = res.body && res.body.getReader ? res.body.getReader() : null;
    if (!reader) {
      throw new Error('stream_not_supported');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalResult = null;
    let finalError = '';

    function handleChunk(chunkText) {
      let eventName = 'message';
      const dataLines = [];
      const lines = String(chunkText || '').split('\n');
      for (const lineRaw of lines) {
        const line = lineRaw.trim();
        if (!line) continue;
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
          continue;
        }
        if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trim());
        }
      }
      if (!dataLines.length) return;
      let eventPayload = null;
      try {
        eventPayload = JSON.parse(dataLines.join('\n'));
      } catch (e) {
        return;
      }
      if (eventName === 'progress') {
        if (onProgress && typeof onProgress === 'function') {
          onProgress(eventPayload || {});
        }
      } else if (eventName === 'result') {
        finalResult = eventPayload || {};
      } else if (eventName === 'error') {
        finalError = String((eventPayload && eventPayload.message) || 'workbench_edit_failed');
      }
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const idx = buffer.indexOf('\n\n');
        if (idx < 0) break;
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleChunk(block);
      }
    }

    if (buffer.trim()) {
      handleChunk(buffer);
      buffer = '';
    }

    if (finalError) {
      throw new Error(finalError);
    }
    if (finalResult) {
      return finalResult;
    }
    throw new Error('workbench_edit_stream_empty_result');
  }

  async function ensurePublicAuth() {
    if (typeof ensurePublicKey !== 'function') {
      throw new Error('鉴权脚本未加载');
    }
    const authHeader = await ensurePublicKey();
    if (authHeader === null) {
      throw new Error('请先登录公开页面');
    }
    return authHeader;
  }

  async function runEdit() {
    if (state.editing) {
      if (workbenchEditAbortController) {
        workbenchEditAbortController.abort();
      }
      return;
    }

    const prompt = String(editPromptInput ? editPromptInput.value : '').trim();
    if (!prompt) {
      toast('请输入编辑提示词', 'warning');
      return;
    }

    if (!state.currentParentPostId && !state.referenceImages.length) {
      toast('请先添加参考图', 'warning');
      return;
    }

    let authHeader = '';
    try {
      authHeader = await ensurePublicAuth();
    } catch (e) {
      toast(String(e.message || e), 'error');
      if (String(e.message || '').includes('登录')) {
        window.location.href = '/login';
      }
      return;
    }

    const body = {
      prompt,
    };

    const references = state.referenceImages
      .slice(0, REFERENCE_LIMIT)
      .map((item) => String(item.data || '').trim())
      .filter(Boolean);
    if (references.length) {
      body.image_references = references;
      const firstRef = references[0];
      if (firstRef.startsWith('data:image/')) {
        body.image_base64 = firstRef;
      } else {
        body.image_url = firstRef;
      }
    }

    if (state.currentParentPostId) {
      body.parent_post_id = state.currentParentPostId;
      if (state.currentSourceImageUrl) {
        body.source_image_url = state.currentSourceImageUrl;
      }
    } else if (!references.length) {
      toast('请先添加参考图', 'warning');
      return;
    }

    setEditing(true);
    forceSubmitButtonAbortClickable();
    startEditProgress();
    workbenchEditAbortController = new AbortController();

    try {
      const payload = await requestWorkbenchEditStream(authHeader, body, (evt) => {
        forceSubmitButtonAbortClickable();
        const next = Number(evt && evt.progress ? evt.progress : 0);
        const message = String((evt && evt.message) || '').trim();
        if (Number.isFinite(next) && next > 0) {
          const safe = Math.max(editProgressValue, Math.min(99, next));
          setEditProgress(safe, message ? `${message} · ${safe}%` : `编辑中 ${safe}%`);
          setStatus('running', message || `编辑中 ${safe}%`);
        } else if (message) {
          setEditProgress(editProgressValue, message);
          setStatus('running', message);
        }
      }, workbenchEditAbortController ? workbenchEditAbortController.signal : undefined);
      const imageUrls = Array.isArray(payload?.data)
        ? payload.data
          .map((item) => String((item && item.url) || '').trim())
          .filter(Boolean)
        : [];
      const imageUrl = imageUrls[0] || '';
      if (!imageUrl) {
        throw new Error('返回结果缺少图片 URL');
      }

      const previousParentPostId = state.currentParentPostId;
      const generatedParent = String(
        payload.current_parent_post_id
        || payload.generated_parent_post_id
        || payload.input_parent_post_id
        || extractParentPostId(imageUrl)
      ).trim();
      const resolvedParentPostId = generatedParent || previousParentPostId;

      const sourceImageUrl = pickSourceImageUrl(
        {
          current_source_image_url: payload.current_source_image_url,
          source_image_url: payload.source_image_url,
          imageUrl,
        },
        resolvedParentPostId,
        state.currentSourceImageUrl
      );

      const mode = String(payload.mode || (state.currentParentPostId ? 'parent_post' : 'upload')).trim();
      const elapsedMs = Number(payload.elapsed_ms || 0);

      state.currentParentPostId = resolvedParentPostId;
      state.currentSourceImageUrl = sourceImageUrl;
      state.currentModeValue = mode || 'parent_post';
      if (parentPostInput) {
        parentPostInput.value = state.currentParentPostId || '';
      }
      setPreviewImages(imageUrls);
      updateMeta();

      state.editRound += 1;
      const createdAt = Date.now();
      const historyEntries = imageUrls.map((url, index) => {
        const perParentPostId = String(extractParentPostId(url) || resolvedParentPostId || '').trim();
        return {
          id: `${createdAt}_${index}_${Math.random().toString(16).slice(2, 8)}`,
          round: state.editRound,
          roundIndex: index + 1,
          roundTotal: imageUrls.length,
          prompt,
          mode: state.currentModeValue,
          imageUrl: url,
          imageUrls,
          parentPostId: perParentPostId,
          sourceImageUrl: String(url || sourceImageUrl || '').trim(),
          elapsedMs: Number.isFinite(elapsedMs) ? Math.max(0, Math.round(elapsedMs)) : 0,
          createdAt,
        };
      });
      state.history = [...historyEntries, ...state.history];
      renderHistory();
      historyEntries.forEach((entry) => {
        rememberParentPost({
          parentPostId: entry.parentPostId || resolvedParentPostId,
          sourceImageUrl: entry.sourceImageUrl || sourceImageUrl,
          imageUrl: entry.imageUrl || imageUrl,
          origin: 'workbench_edit',
        });
      });

      finishEditProgress(true, '编辑完成 100%');
      setStatus('done', `编辑完成 · round #${state.editRound}（${imageUrls.length} 张）`);
      toast('编辑成功，已更新当前画面', 'success');
    } catch (e) {
      if (e && e.name === 'AbortError') {
        finishEditProgress(false, '已中止');
        setStatus('error', '已中止');
        toast('已中止编辑', 'warning');
      } else {
        finishEditProgress(false, '编辑失败');
        setStatus('error', '编辑失败');
        toast(String(e.message || e), 'error');
      }
    } finally {
      workbenchEditAbortController = null;
      setEditing(false);
    }
  }

  function bindEvents() {
    if (selectSeedBtn && seedImageInput) {
      selectSeedBtn.addEventListener('click', () => {
        seedImageInput.click();
      });

      seedImageInput.addEventListener('change', async (event) => {
        const files = event.target && event.target.files ? Array.from(event.target.files) : [];
        if (!files.length) return;

        try {
          const added = await addReferenceFiles(files, 'upload');
          if (added > 0) {
            toast(`已添加 ${added} 张参考图`, 'success');
          }
        } catch (e) {
          toast(String(e.message || e), 'error');
        } finally {
          seedImageInput.value = '';
        }
      });
    }

    if (submitEditBtn) {
      submitEditBtn.addEventListener('click', runEdit);
    }

    if (applyParentBtn) {
      applyParentBtn.addEventListener('click', async () => {
        await applyParentPostFromText(parentPostInput ? parentPostInput.value : '', { addToReferences: true });
      });
    }

    if (parentPostInput) {
      parentPostInput.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          await applyParentPostFromText(parentPostInput.value, { addToReferences: true });
        }
      });
      parentPostInput.addEventListener('input', async () => {
        const raw = parentPostInput.value.trim();
        if (!raw) return;
        await applyParentPostFromText(raw, { silent: true });
      });
      parentPostInput.addEventListener('paste', (event) => {
        const text = String(event.clipboardData ? event.clipboardData.getData('text') || '' : '').trim();
        if (!text) return;
        event.preventDefault();
        parentPostInput.value = text;
        applyParentPostFromText(text, { silent: true });
      });
    }

    if (editPromptInput) {
      editPromptInput.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
          event.preventDefault();
          runEdit();
        }
      });
    }

    if (resetCycleBtn) {
      resetCycleBtn.addEventListener('click', () => {
        resetCycle(true);
        toast('已重置编辑链路', 'success');
      });
    }

    if (clearHistoryBtn) {
      clearHistoryBtn.addEventListener('click', () => {
        clearHistory();
        toast('历史记录已清空', 'success');
      });
    }

    document.addEventListener('paste', async (event) => {
      const dataTransfer = event.clipboardData;
      const files = pickImageFilesFromDataTransfer(dataTransfer);
      if (files.length) {
        event.preventDefault();
        try {
          const added = await addReferenceFiles(files, 'paste');
          if (added > 0) {
            toast(`已粘贴 ${added} 张参考图`, 'success');
          }
        } catch (e) {
          toast(String(e.message || e), 'error');
        }
        return;
      }
      const text = dataTransfer ? String(dataTransfer.getData('text') || '').trim() : '';
      if (!text) return;
      const target = event.target;
      const isTypingInPrompt = target === editPromptInput;
      const isTypingInParentInput = target === parentPostInput;
      if (isTypingInPrompt) return;
      if (!isTypingInParentInput && (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) {
        return;
      }
      const ok = applyParentPostFromText(text, { silent: true });
      if (ok) {
        event.preventDefault();
      }
    });

    if (previewShell) {
      previewShell.addEventListener('dragenter', (event) => {
        if (!hasFiles(event.dataTransfer)) return;
        event.preventDefault();
        dragCounter += 1;
        setDragActive(true);
      });

      previewShell.addEventListener('dragover', (event) => {
        if (!hasFiles(event.dataTransfer)) return;
        event.preventDefault();
        if (event.dataTransfer) {
          event.dataTransfer.dropEffect = 'copy';
        }
        setDragActive(true);
      });

      previewShell.addEventListener('dragleave', (event) => {
        if (!hasFiles(event.dataTransfer)) return;
        event.preventDefault();
        dragCounter = Math.max(0, dragCounter - 1);
        if (dragCounter === 0) {
          setDragActive(false);
        }
      });

      previewShell.addEventListener('drop', async (event) => {
        event.preventDefault();
        dragCounter = 0;
        setDragActive(false);
        const files = pickImageFilesFromDataTransfer(event.dataTransfer);
        if (!files.length) {
          toast('未检测到可用图片文件', 'warning');
          return;
        }
        try {
          const added = await addReferenceFiles(files, 'drop');
          if (added > 0) {
            toast(`已拖入 ${added} 张参考图`, 'success');
          }
        } catch (e) {
          toast(String(e.message || e), 'error');
        }
      });
    }

    window.addEventListener('dragover', (event) => {
      if (!hasFiles(event.dataTransfer)) return;
      event.preventDefault();
    });

    window.addEventListener('drop', (event) => {
      if (!hasFiles(event.dataTransfer)) return;
      if (previewShell && previewShell.contains(event.target)) {
        return;
      }
      event.preventDefault();
      setDragActive(false);
      dragCounter = 0;
    });
  }

  function mountInlineSubmitButton() {
    if (!editPromptInput || !submitEditBtn) return false;
    const promptWrap = editPromptInput.closest('.prompt-enhance-wrap');
    if (!promptWrap) return false;
    if (!promptWrap.contains(submitEditBtn)) {
      const actionRow = submitEditBtn.parentElement;
      promptWrap.appendChild(submitEditBtn);
      if (actionRow && actionRow.classList && actionRow.classList.contains('action-row')) {
        actionRow.classList.add('action-row-inline');
      }
    }
    submitEditBtn.classList.add('inline-submit-btn');
    return true;
  }

  function ensureInlineSubmitButton(attempt = 0) {
    if (mountInlineSubmitButton()) return;
    if (attempt >= 20) return;
    setTimeout(() => ensureInlineSubmitButton(attempt + 1), 50);
  }

  function init() {
    ensureInlineSubmitButton();
    bindEvents();
    renderReferenceStrip();
    renderHistory();
    resetCycle(false);
    updateMeta();
  }

  init();
})();
