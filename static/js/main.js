document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('files');
    const fileList = document.getElementById('file-list');
    const analyzeBtn = document.getElementById('analyze-btn');

    if (fileInput && fileList && analyzeBtn) {
        fileInput.addEventListener('change', () => {
            fileList.innerHTML = '';
            const files = Array.from(fileInput.files);
            files.forEach(file => {
                const li = document.createElement('li');
                li.textContent = file.name;
                li.className = 'p-3 bg-gray-700 rounded-lg shadow-sm hover:bg-gray-600 transition';
                fileList.appendChild(li);
            });
            analyzeBtn.disabled = files.length < 2;
        });
    }
});

function showCode(code1, code2) {
    document.getElementById('code1').innerHTML = code1;
    document.getElementById('code2').innerHTML = code2;
    document.getElementById('code-modal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('code-modal').classList.add('hidden');
}