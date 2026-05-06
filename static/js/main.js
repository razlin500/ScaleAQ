document.addEventListener('DOMContentLoaded', () => {

    // ============================================================
    // SCROLLYTELLING: reveal panels on scroll
    // ============================================================
    const storyPanels = document.querySelectorAll('.panel-raw, .panel-annotated');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.3,
    });

    storyPanels.forEach(panel => observer.observe(panel));

    // ============================================================
    // GALLERY: click to detect
    // ============================================================
    const thumbs = document.querySelectorAll('.gallery-thumb');
    const resultPanel = document.getElementById('result-panel');
    const resultContent = document.getElementById('result-content');

    thumbs.forEach(thumb => {
        thumb.addEventListener('click', () => {
            const filename = thumb.dataset.filename;

            // Highlight selected
            thumbs.forEach(t => t.classList.remove('selected'));
            thumb.classList.add('selected');

            // Show loading spinner
            resultPanel.style.display = 'block';
            resultContent.innerHTML = `
                <div class="loader-container">
                    <div class="spinner"></div>
                    <p>Analyzing net image…</p>
                </div>`;
            resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

            fetch('/api/detect_path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: filename }),
            })
            .then(res => {
                if (!res.ok) throw new Error('Detection failed');
                return res.json();
            })
            .then(data => {
                if (data.error) throw new Error(data.error);
                const holeWord = data.count === 1 ? 'Hole' : 'Holes';
                resultContent.innerHTML = `
                    <div class="result-card">
                        <h3>${filename}</h3>
                        <div class="stat-badge">${data.count} ${holeWord} Detected</div>
                        <div>
                            <img src="${data.annotated_image}" alt="Detection result for ${filename}">
                        </div>
                    </div>`;
            })
            .catch(err => {
                resultContent.innerHTML = `
                    <div class="result-card">
                        <p style="color:#ef4444;">Error: ${err.message}</p>
                    </div>`;
            });
        });
    });
});
