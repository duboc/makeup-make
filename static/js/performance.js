// Performance optimization utilities for the MAC Foundation Color Matcher

// Lazy loading for images
class LazyImageLoader {
    constructor() {
        this.imageObserver = null;
        this.init();
    }

    init() {
        if ('IntersectionObserver' in window) {
            this.imageObserver = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        this.loadImage(img);
                        observer.unobserve(img);
                    }
                });
            }, {
                rootMargin: '50px 0px',
                threshold: 0.01
            });
        }
    }

    loadImage(img) {
        const src = img.dataset.src;
        if (!src) return;

        // Create a new image to preload
        const tempImg = new Image();
        tempImg.onload = () => {
            img.src = src;
            img.classList.add('loaded');
            delete img.dataset.src;
        };
        tempImg.src = src;
    }

    observe(images) {
        if (!this.imageObserver) {
            // Fallback for browsers without IntersectionObserver
            images.forEach(img => this.loadImage(img));
            return;
        }

        images.forEach(img => {
            this.imageObserver.observe(img);
        });
    }
}

// Virtual scrolling for large result sets
class VirtualScroller {
    constructor(container, itemHeight, renderItem) {
        this.container = container;
        this.itemHeight = itemHeight;
        this.renderItem = renderItem;
        this.items = [];
        this.scrollTop = 0;
        this.visibleStart = 0;
        this.visibleEnd = 0;
        this.displayContainer = null;
        
        this.init();
    }

    init() {
        // Create display container
        this.displayContainer = document.createElement('div');
        this.displayContainer.style.position = 'relative';
        this.container.appendChild(this.displayContainer);

        // Set up scroll listener
        this.container.addEventListener('scroll', this.handleScroll.bind(this));
        this.container.style.overflow = 'auto';
        this.container.style.position = 'relative';
    }

    setItems(items) {
        this.items = items;
        this.container.style.height = `${items.length * this.itemHeight}px`;
        this.render();
    }

    handleScroll() {
        this.scrollTop = this.container.scrollTop;
        this.render();
    }

    render() {
        const containerHeight = this.container.clientHeight;
        const visibleStart = Math.floor(this.scrollTop / this.itemHeight);
        const visibleEnd = Math.ceil((this.scrollTop + containerHeight) / this.itemHeight);

        // Only update if visible range changed
        if (visibleStart !== this.visibleStart || visibleEnd !== this.visibleEnd) {
            this.visibleStart = visibleStart;
            this.visibleEnd = visibleEnd;

            // Clear container
            this.displayContainer.innerHTML = '';

            // Render visible items
            for (let i = visibleStart; i < visibleEnd && i < this.items.length; i++) {
                const itemElement = this.renderItem(this.items[i], i);
                itemElement.style.position = 'absolute';
                itemElement.style.top = `${i * this.itemHeight}px`;
                itemElement.style.left = '0';
                itemElement.style.right = '0';
                this.displayContainer.appendChild(itemElement);
            }
        }
    }
}

// Request Animation Frame throttling
class RAFThrottle {
    constructor(callback) {
        this.callback = callback;
        this.requestId = null;
        this.lastArgs = null;
    }

    call(...args) {
        this.lastArgs = args;
        if (!this.requestId) {
            this.requestId = requestAnimationFrame(() => {
                this.callback(...this.lastArgs);
                this.requestId = null;
            });
        }
    }

    cancel() {
        if (this.requestId) {
            cancelAnimationFrame(this.requestId);
            this.requestId = null;
        }
    }
}

// Optimized color swatch renderer
class ColorSwatchRenderer {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.cache = new Map();
    }

    createSwatch(labColor, size = { width: 100, height: 50 }) {
        const cacheKey = `${labColor.join(',')}_${size.width}x${size.height}`;
        
        // Check cache first
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        // Set canvas size
        this.canvas.width = size.width;
        this.canvas.height = size.height;

        // Convert LAB to RGB (simplified for performance)
        const rgb = this.labToRgb(labColor);
        
        // Draw swatch with gradient effect
        const gradient = this.ctx.createLinearGradient(0, 0, 0, size.height);
        gradient.addColorStop(0, `rgba(${rgb.join(',')}, 0.9)`);
        gradient.addColorStop(1, `rgba(${rgb.join(',')}, 1)`);
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, size.width, size.height);

        // Add subtle border
        this.ctx.strokeStyle = 'rgba(0,0,0,0.1)';
        this.ctx.strokeRect(0.5, 0.5, size.width - 1, size.height - 1);

        // Convert to data URL and cache
        const dataUrl = this.canvas.toDataURL('image/png');
        this.cache.set(cacheKey, dataUrl);

        // Limit cache size
        if (this.cache.size > 100) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        return dataUrl;
    }

    labToRgb(lab) {
        // Simplified LAB to RGB conversion
        const [L, a, b] = lab;
        const L_norm = L / 100;
        const a_norm = a / 127;
        const b_norm = b / 127;
        
        const r = Math.round(Math.max(0, Math.min(255, (L_norm + 0.5 * a_norm) * 255)));
        const g = Math.round(Math.max(0, Math.min(255, (L_norm - 0.5 * a_norm + 0.5 * b_norm) * 255)));
        const b_rgb = Math.round(Math.max(0, Math.min(255, (L_norm - 0.5 * b_norm) * 255)));
        
        return [r, g, b_rgb];
    }
}

// Web Worker for heavy computations
class ColorMatchWorker {
    constructor() {
        this.worker = null;
        this.initWorker();
    }

    initWorker() {
        const workerCode = `
            self.addEventListener('message', function(e) {
                const { type, data } = e.data;
                
                switch(type) {
                    case 'calculateDeltaE':
                        const deltaE = calculateDeltaE(data.lab1, data.lab2);
                        self.postMessage({ type: 'deltaE', result: deltaE });
                        break;
                        
                    case 'sortMatches':
                        const sorted = data.matches.sort((a, b) => a.deltaE - b.deltaE);
                        self.postMessage({ type: 'sorted', result: sorted });
                        break;
                }
            });
            
            function calculateDeltaE(lab1, lab2) {
                return Math.sqrt(
                    Math.pow(lab1[0] - lab2[0], 2) +
                    Math.pow(lab1[1] - lab2[1], 2) +
                    Math.pow(lab1[2] - lab2[2], 2)
                );
            }
        `;

        const blob = new Blob([workerCode], { type: 'application/javascript' });
        this.worker = new Worker(URL.createObjectURL(blob));
    }

    calculateDeltaE(lab1, lab2) {
        return new Promise((resolve) => {
            this.worker.postMessage({ type: 'calculateDeltaE', data: { lab1, lab2 } });
            this.worker.onmessage = (e) => {
                if (e.data.type === 'deltaE') {
                    resolve(e.data.result);
                }
            };
        });
    }

    sortMatches(matches) {
        return new Promise((resolve) => {
            this.worker.postMessage({ type: 'sortMatches', data: { matches } });
            this.worker.onmessage = (e) => {
                if (e.data.type === 'sorted') {
                    resolve(e.data.result);
                }
            };
        });
    }

    terminate() {
        if (this.worker) {
            this.worker.terminate();
        }
    }
}

// Export utilities
window.PerformanceUtils = {
    LazyImageLoader,
    VirtualScroller,
    RAFThrottle,
    ColorSwatchRenderer,
    ColorMatchWorker
};
