// Enhanced results page functionality with filtering and performance optimizations

class ResultsEnhancer {
    constructor() {
        this.results = [];
        this.filteredResults = [];
        this.filters = {
            productLine: 'all',
            undertone: 'all',
            coverage: 'all',
            searchTerm: '',
            minMatch: 0
        };
        
        this.swatchRenderer = new window.PerformanceUtils.ColorSwatchRenderer();
        this.lazyLoader = new window.PerformanceUtils.LazyImageLoader();
        
        this.init();
    }

    init() {
        this.setupFilterControls();
        this.setupSearch();
        this.setupSorting();
        this.enhanceResultCards();
        this.addComparisonFeature();
        this.setupKeyboardNavigation();
    }

    setupFilterControls() {
        // Create filter UI
        const filterContainer = document.createElement('div');
        filterContainer.className = 'filter-container mb-4 p-3 bg-light rounded shadow-sm';
        filterContainer.innerHTML = `
            <div class="row g-3">
                <div class="col-md-3">
                    <label class="form-label small fw-bold">Product Line</label>
                    <select class="form-select form-select-sm" id="filter-product-line">
                        <option value="all">All Products</option>
                        <option value="Studio Fix Fluid">Studio Fix Fluid</option>
                        <option value="Face and Body">Face and Body</option>
                        <option value="Pro Longwear">Pro Longwear</option>
                        <option value="Studio Radiance">Studio Radiance</option>
                        <option value="Next to Nothing">Next to Nothing</option>
                        <option value="Lightful C³">Lightful C³ Powder</option>
                    </select>
                </div>
                <div class="col-md-3">
                    <label class="form-label small fw-bold">Undertone</label>
                    <select class="form-select form-select-sm" id="filter-undertone">
                        <option value="all">All Undertones</option>
                        <option value="Cool">Cool (Pink)</option>
                        <option value="Warm">Warm (Yellow)</option>
                        <option value="Neutral">Neutral</option>
                    </select>
                </div>
                <div class="col-md-3">
                    <label class="form-label small fw-bold">Coverage</label>
                    <select class="form-select form-select-sm" id="filter-coverage">
                        <option value="all">All Coverage</option>
                        <option value="Sheer">Sheer</option>
                        <option value="Light">Light</option>
                        <option value="Light to Medium">Light to Medium</option>
                        <option value="Medium">Medium</option>
                        <option value="Full">Full</option>
                    </select>
                </div>
                <div class="col-md-3">
                    <label class="form-label small fw-bold">Min Match %</label>
                    <div class="input-group input-group-sm">
                        <input type="range" class="form-range" id="filter-match-score" 
                               min="0" max="100" value="0" step="5">
                        <span class="input-group-text" id="match-score-display">0%</span>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-md-6">
                    <div class="input-group input-group-sm">
                        <span class="input-group-text"><i class="fas fa-search"></i></span>
                        <input type="text" class="form-control" id="filter-search" 
                               placeholder="Search by shade name (e.g., NC25, NW20)">
                    </div>
                </div>
                <div class="col-md-6 text-end">
                    <button class="btn btn-sm btn-outline-secondary" id="reset-filters">
                        <i class="fas fa-undo me-1"></i>Reset Filters
                    </button>
                    <span class="ms-3 text-muted small">
                        Showing <span id="filter-count">0</span> of <span id="total-count">0</span> matches
                    </span>
                </div>
            </div>
        `;

        const resultsContainer = document.querySelector('.results-container');
        if (resultsContainer) {
            resultsContainer.insertBefore(filterContainer, resultsContainer.firstChild);
        }

        // Add event listeners
        document.getElementById('filter-product-line').addEventListener('change', (e) => {
            this.filters.productLine = e.target.value;
            this.applyFilters();
        });

        document.getElementById('filter-undertone').addEventListener('change', (e) => {
            this.filters.undertone = e.target.value;
            this.applyFilters();
        });

        document.getElementById('filter-coverage').addEventListener('change', (e) => {
            this.filters.coverage = e.target.value;
            this.applyFilters();
        });

        const matchScoreSlider = document.getElementById('filter-match-score');
        const matchScoreDisplay = document.getElementById('match-score-display');
        
        matchScoreSlider.addEventListener('input', (e) => {
            this.filters.minMatch = parseInt(e.target.value);
            matchScoreDisplay.textContent = `${this.filters.minMatch}%`;
            this.applyFilters();
        });

        document.getElementById('reset-filters').addEventListener('click', () => {
            this.resetFilters();
        });
    }

    setupSearch() {
        const searchInput = document.getElementById('filter-search');
        if (searchInput) {
            // Debounced search
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.filters.searchTerm = e.target.value.toLowerCase();
                    this.applyFilters();
                }, 300);
            });
        }
    }

    setupSorting() {
        // Add sorting options
        const sortContainer = document.createElement('div');
        sortContainer.className = 'sort-container mb-3';
        sortContainer.innerHTML = `
            <div class="btn-group btn-group-sm" role="group">
                <span class="btn btn-outline-secondary disabled">Sort by:</span>
                <button type="button" class="btn btn-outline-primary active" data-sort="match">
                    Best Match
                </button>
                <button type="button" class="btn btn-outline-primary" data-sort="shade">
                    Shade Name
                </button>
                <button type="button" class="btn btn-outline-primary" data-sort="lightness">
                    Light to Dark
                </button>
                <button type="button" class="btn btn-outline-primary" data-sort="undertone">
                    Undertone
                </button>
            </div>
        `;

        const filterContainer = document.querySelector('.filter-container');
        if (filterContainer) {
            filterContainer.appendChild(sortContainer);
        }

        // Add sorting event listeners
        sortContainer.addEventListener('click', (e) => {
            if (e.target.dataset.sort) {
                // Update active state
                sortContainer.querySelectorAll('.btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                e.target.classList.add('active');

                // Apply sort
                this.sortResults(e.target.dataset.sort);
            }
        });
    }

    applyFilters() {
        this.filteredResults = this.results.filter(result => {
            // Product line filter
            if (this.filters.productLine !== 'all' && 
                result.foundation.product_line !== this.filters.productLine) {
                return false;
            }

            // Undertone filter
            if (this.filters.undertone !== 'all' && 
                result.foundation.undertone !== this.filters.undertone) {
                return false;
            }

            // Coverage filter
            if (this.filters.coverage !== 'all' && 
                result.foundation.coverage !== this.filters.coverage) {
                return false;
            }

            // Match score filter
            if (result.match_score < this.filters.minMatch) {
                return false;
            }

            // Search filter
            if (this.filters.searchTerm) {
                const searchableText = `${result.foundation.shade} ${result.foundation.description}`.toLowerCase();
                if (!searchableText.includes(this.filters.searchTerm)) {
                    return false;
                }
            }

            return true;
        });

        this.updateDisplay();
    }

    sortResults(sortBy) {
        switch (sortBy) {
            case 'match':
                this.filteredResults.sort((a, b) => b.match_score - a.match_score);
                break;
            case 'shade':
                this.filteredResults.sort((a, b) => 
                    a.foundation.shade.localeCompare(b.foundation.shade));
                break;
            case 'lightness':
                this.filteredResults.sort((a, b) => b.foundation.L - a.foundation.L);
                break;
            case 'undertone':
                this.filteredResults.sort((a, b) => 
                    a.foundation.undertone.localeCompare(b.foundation.undertone));
                break;
        }
        this.updateDisplay();
    }

    resetFilters() {
        this.filters = {
            productLine: 'all',
            undertone: 'all',
            coverage: 'all',
            searchTerm: '',
            minMatch: 0
        };

        // Reset UI
        document.getElementById('filter-product-line').value = 'all';
        document.getElementById('filter-undertone').value = 'all';
        document.getElementById('filter-coverage').value = 'all';
        document.getElementById('filter-search').value = '';
        document.getElementById('filter-match-score').value = 0;
        document.getElementById('match-score-display').textContent = '0%';

        this.applyFilters();
    }

    updateDisplay() {
        // Update counts
        document.getElementById('filter-count').textContent = this.filteredResults.length;
        document.getElementById('total-count').textContent = this.results.length;

        // Re-render results
        const resultsGrid = document.querySelector('.results-grid');
        if (resultsGrid) {
            // Clear existing results
            resultsGrid.innerHTML = '';

            // Add filtered results with fade-in animation
            this.filteredResults.forEach((result, index) => {
                const card = this.createResultCard(result);
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                resultsGrid.appendChild(card);

                // Animate in
                setTimeout(() => {
                    card.style.transition = 'all 0.3s ease';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 50);
            });

            // Show empty state if no results
            if (this.filteredResults.length === 0) {
                resultsGrid.innerHTML = `
                    <div class="col-12 text-center py-5">
                        <i class="fas fa-search fa-3x text-muted mb-3"></i>
                        <h5 class="text-muted">No matches found</h5>
                        <p class="text-muted">Try adjusting your filters or search criteria</p>
                    </div>
                `;
            }

            // Apply lazy loading to new images
            setTimeout(() => {
                const lazyImages = resultsGrid.querySelectorAll('.lazy-load');
                this.lazyLoader.observe(lazyImages);
            }, 100);
        }
    }

    createResultCard(result) {
        const card = document.createElement('div');
        card.className = 'col-md-6 col-lg-4 mb-4';
        card.innerHTML = `
            <div class="card h-100 result-card shadow-sm hover-shadow" data-shade="${result.foundation.shade}">
                <div class="card-body">
                    <div class="d-flex align-items-start mb-3">
                        <img class="color-swatch me-3" 
                             src="data:image/png;base64,${result.swatch_b64}"
                             width="60" height="40" 
                             alt="${result.foundation.shade} swatch">
                        <div class="flex-grow-1">
                            <h5 class="card-title mb-1">${result.foundation.shade}</h5>
                            <p class="text-muted small mb-0">${result.foundation.product_line}</p>
                        </div>
                        <div class="text-end">
                            <div class="match-score-circle ${result.match_class}">
                                ${Math.round(result.match_score)}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="shade-details">
                        <div class="row g-2 mb-2">
                            <div class="col-6">
                                <span class="badge bg-light text-dark w-100">
                                    <i class="fas fa-palette me-1"></i>${result.foundation.undertone}
                                </span>
                            </div>
                            <div class="col-6">
                                <span class="badge bg-light text-dark w-100">
                                    <i class="fas fa-layer-group me-1"></i>${result.foundation.coverage}
                                </span>
                            </div>
                        </div>
                        
                        <p class="small text-muted mb-2">${result.foundation.description}</p>
                        
                        <div class="color-values small">
                            <span class="me-2">L*: ${result.foundation.L.toFixed(1)}</span>
                            <span class="me-2">a*: ${result.foundation.a.toFixed(1)}</span>
                            <span>b*: ${result.foundation.b.toFixed(1)}</span>
                        </div>
                    </div>
                    
                    <div class="mt-3 d-grid gap-2">
                        <button class="btn btn-sm btn-outline-primary compare-btn" 
                                data-shade="${result.foundation.shade}">
                            <i class="fas fa-exchange-alt me-1"></i>Compare
                        </button>
                    </div>
                </div>
            </div>
        `;

        return card;
    }

    enhanceResultCards() {
        // Add hover effects and interactions
        document.addEventListener('mouseover', (e) => {
            if (e.target.closest('.result-card')) {
                const card = e.target.closest('.result-card');
                card.style.transform = 'translateY(-5px)';
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.closest('.result-card')) {
                const card = e.target.closest('.result-card');
                card.style.transform = 'translateY(0)';
            }
        });
    }

    addComparisonFeature() {
        const compareContainer = document.createElement('div');
        compareContainer.className = 'comparison-container';
        compareContainer.innerHTML = `
            <div class="position-fixed bottom-0 end-0 p-3" style="z-index: 1000;">
                <div class="comparison-panel bg-white rounded shadow-lg p-3" style="display: none; width: 350px;">
                    <h6 class="mb-3">Compare Shades</h6>
                    <div class="comparison-slots"></div>
                    <button class="btn btn-sm btn-outline-secondary mt-3 w-100" id="clear-comparison">
                        Clear Comparison
                    </button>
                </div>
                <button class="btn btn-primary rounded-circle shadow-lg comparison-toggle" 
                        style="width: 60px; height: 60px;">
                    <i class="fas fa-exchange-alt"></i>
                    <span class="badge bg-danger position-absolute top-0 start-100 translate-middle" 
                          id="comparison-count" style="display: none;">0</span>
                </button>
            </div>
        `;
        document.body.appendChild(compareContainer);

        this.setupComparisonListeners();
    }

    setupComparisonListeners() {
        const comparisonShades = new Set();
        const maxComparison = 3;

        document.addEventListener('click', (e) => {
            if (e.target.closest('.compare-btn')) {
                const shade = e.target.closest('.compare-btn').dataset.shade;
                
                if (comparisonShades.has(shade)) {
                    comparisonShades.delete(shade);
                } else if (comparisonShades.size < maxComparison) {
                    comparisonShades.add(shade);
                } else {
                    alert(`You can compare up to ${maxComparison} shades at once`);
                    return;
                }

                this.updateComparisonPanel(comparisonShades);
            }

            if (e.target.closest('.comparison-toggle')) {
                const panel = document.querySelector('.comparison-panel');
                panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            }

            if (e.target.id === 'clear-comparison') {
                comparisonShades.clear();
                this.updateComparisonPanel(comparisonShades);
            }
        });
    }

    updateComparisonPanel(shades) {
        const count = shades.size;
        const badge = document.getElementById('comparison-count');
        const slotsContainer = document.querySelector('.comparison-slots');

        // Update badge
        badge.textContent = count;
        badge.style.display = count > 0 ? 'inline-block' : 'none';

        // Update comparison slots
        slotsContainer.innerHTML = '';
        
        if (count === 0) {
            slotsContainer.innerHTML = '<p class="text-muted text-center">No shades selected for comparison</p>';
            return;
        }

        const selectedResults = this.results.filter(r => shades.has(r.foundation.shade));
        
        selectedResults.forEach(result => {
            const slot = document.createElement('div');
            slot.className = 'comparison-slot mb-2 p-2 border rounded';
            slot.innerHTML = `
                <div class="d-flex align-items-center">
                    <img src="data:image/png;base64,${result.swatch_b64}" 
                         width="40" height="30" class="me-2">
                    <div class="flex-grow-1">
                        <strong>${result.foundation.shade}</strong>
                        <br>
                        <small class="text-muted">${result.match_score}% match</small>
                    </div>
                    <button class="btn btn-sm btn-link text-danger" 
                            onclick="this.closest('.comparison-slot').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            slotsContainer.appendChild(slot);
        });

        // Add comparison summary
        if (count > 1) {
            const summary = document.createElement('div');
            summary.className = 'comparison-summary mt-3 p-2 bg-light rounded';
            summary.innerHTML = `
                <small class="text-muted">
                    <strong>Comparison:</strong><br>
                    Lightest: ${this.getLightestShade(selectedResults)}<br>
                    Most similar: ${this.getMostSimilarPair(selectedResults)}
                </small>
            `;
            slotsContainer.appendChild(summary);
        }
    }

    getLightestShade(results) {
        return results.reduce((lightest, current) => 
            current.foundation.L > lightest.foundation.L ? current : lightest
        ).foundation.shade;
    }

    getMostSimilarPair(results) {
        let minDelta = Infinity;
        let pair = '';

        for (let i = 0; i < results.length; i++) {
            for (let j = i + 1; j < results.length; j++) {
                const delta = Math.sqrt(
                    Math.pow(results[i].foundation.L - results[j].foundation.L, 2) +
                    Math.pow(results[i].foundation.a - results[j].foundation.a, 2) +
                    Math.pow(results[i].foundation.b - results[j].foundation.b, 2)
                );
                
                if (delta < minDelta) {
                    minDelta = delta;
                    pair = `${results[i].foundation.shade} & ${results[j].foundation.shade}`;
                }
            }
        }

        return pair;
    }

    setupKeyboardNavigation() {
        let currentFocus = -1;
        const cards = document.querySelectorAll('.result-card');

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                
                if (e.key === 'ArrowDown') {
                    currentFocus = Math.min(currentFocus + 1, cards.length - 1);
                } else {
                    currentFocus = Math.max(currentFocus - 1, 0);
                }

                cards.forEach((card, index) => {
                    card.classList.toggle('focused', index === currentFocus);
                });

                if (cards[currentFocus]) {
                    cards[currentFocus].scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }

            if (e.key === 'Enter' && currentFocus >= 0) {
                const compareBtn = cards[currentFocus].querySelector('.compare-btn');
                if (compareBtn) {
                    compareBtn.click();
                }
            }
        });
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('.results-container')) {
        window.resultsEnhancer = new ResultsEnhancer();
    }
});
