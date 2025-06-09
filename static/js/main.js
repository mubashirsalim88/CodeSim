document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('files');
    const fileList = document.getElementById('file-list');
    const analyzeBtn = document.getElementById('analyze-btn');

    if (fileInput && fileList && analyzeBtn) {
        fileInput.addEventListener('change', (event) => {
            fileList.innerHTML = '';
            const files = Array.from(fileInput.files);
            files.forEach(file => {
                const li = document.createElement('li');
                li.textContent = file.name;
                li.className = 'p-3 bg-gray-600/50 rounded-lg shadow-sm flex items-center space-x-3 hover:bg-gray-500/50 transition duration-200';
                const icon = document.createElement('svg');
                icon.className = 'w-5 h-5 text-indigo-400';
                icon.innerHTML = '<path fill="currentColor" d="M4 4h12v12H4V4zm2 2v8h8V6H6zm2 2h4v4H8V8z"/>';
                li.prepend(icon);
                fileList.appendChild(li);
            });
            analyzeBtn.disabled = files.length < 2;
            analyzeBtn.classList.toggle('opacity-50', files.length < 2);
        });

        // Accessibility: Allow Enter key to trigger file input
        fileInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                fileInput.click();
            }
        });
    }

    // Auto-highlight code with Prism.js when modal is shown
    window.showCode = function(code1, code2, file1, file2) {
        const code1Element = document.getElementById('code1');
        const code2Element = document.getElementById('code2');
        document.getElementById('file1-title').textContent = file1 || 'File 1';
        document.getElementById('file2-title').textContent = file2 || 'File 2';
        code1Element.innerHTML = code1;
        code2Element.innerHTML = code2;
        Prism.highlightElement(code1Element);
        Prism.highlightElement(code2Element);
        const modal = document.getElementById('code-modal');
        modal.classList.remove('hidden');
        modal.classList.add('flex', 'items-center', 'justify-center');
        modal.classList.add('opacity-100');
        modal.focus();
    };

    window.closeModal = function() {
        const modal = document.getElementById('code-modal');
        modal.classList.add('opacity-0');
        setTimeout(() => {
            modal.classList.add('hidden');
            modal.classList.remove('flex', 'items-center', 'justify-center', 'opacity-0');
        }, 300);
    };

    // Close modal with Escape key
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !document.getElementById('code-modal').classList.contains('hidden')) {
            closeModal();
        }
    });
});

// Custom animation for flash messages
const flashMessages = document.querySelectorAll('[role="alert"]');
flashMessages.forEach((msg, index) => {
    msg.style.animationDelay = `${index * 100}ms`;
    setTimeout(() => {
        msg.classList.add('opacity-0');
        setTimeout(() => msg.remove(), 300);
    }, 5000);
});