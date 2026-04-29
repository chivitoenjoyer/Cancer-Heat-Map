(() => {
  const form = document.querySelector('[data-form="predict"]');
  const fileInput = document.querySelector('[data-input="image"]');
  const dropzone = document.querySelector('[data-dropzone]');
  const pickBtn = document.querySelector('[data-action="pick"]');
  const submitBtn = document.querySelector('[data-action="submit"]');
  const previewImg = document.querySelector('[data-preview="img"]');
  const previewName = document.querySelector('[data-preview="name"]');
  const previewHint = document.querySelector('[data-preview="hint"]');

  function setPreview(file) {
    if (!file) {
      previewImg.removeAttribute('src');
      previewImg.style.display = 'none';
      previewName.textContent = 'Ningún archivo seleccionado';
      previewHint.textContent = 'Puedes arrastrar y soltar una imagen aquí o elegir un archivo.';
      return;
    }

    previewImg.style.display = 'block';
    previewName.textContent = file.name;
    previewHint.textContent = `${Math.round(file.size / 1024)} KB · ${file.type || 'image/*'}`;

    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.onload = () => URL.revokeObjectURL(url);
  }

  function setFile(file) {
    if (!fileInput) return;

    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    setPreview(file);
  }

  if (pickBtn && fileInput) {
    pickBtn.addEventListener('click', (e) => {
      e.preventDefault();
      fileInput.click();
    });

    fileInput.addEventListener('change', () => {
      const file = fileInput.files && fileInput.files[0];
      setPreview(file);
    });
  }

  if (dropzone) {
    dropzone.addEventListener('click', () => fileInput && fileInput.click());

    ;['dragenter', 'dragover'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ;['dragleave', 'drop'].forEach((evt) => {
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) setFile(file);
    });
  }

  if (form && submitBtn) {
    form.addEventListener('submit', () => {
      submitBtn.disabled = true;
      const label = submitBtn.querySelector('[data-submit-label]');
      const spin = submitBtn.querySelector('[data-submit-spinner]');
      if (label) label.textContent = 'Procesando…';
      if (spin) spin.style.display = 'inline-block';
    });
  }

  // estado inicial
  if (fileInput && fileInput.files && fileInput.files[0]) {
    setPreview(fileInput.files[0]);
  } else {
    setPreview(null);
  }
})();
