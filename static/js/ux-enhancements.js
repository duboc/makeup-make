/**
 * UX Enhancement Module
 * Provides delightful user experience improvements
 */

class UXEnhancements {
    constructor() {
        this.isFirstVisit = !localStorage.getItem('has_visited');
        this.userPreferences = this.loadPreferences();
        this.hapticEnabled = 'vibrate' in navigator;
        
        this.init();
    }
    
    init() {
        // Initialize onboarding for first-time users
        if (this.isFirstVisit) {
            this.showOnboarding();
        }
        
        // Add microinteractions
        this.setupMicrointeractions();
        
        // Setup success celebrations
        this.setupSuccessCelebrations();
        
        // Initialize tooltips
        this.initTooltips();
        
        // Setup progressive enhancement
        this.setupProgressiveEnhancement();
    }
    
    /**
     * Onboarding Flow
     */
    showOnboarding() {
        const steps = [
            {
                title: 'Bem-vinda ao Foundation Matcher!',
                content: 'Descubra sua base perfeita O Boticário em 3 passos simples',
                image: 'welcome-icon',
                action: 'next'
            },
            {
                title: 'Tire uma Foto',
                content: 'Use luz natural para melhores resultados. Nossa IA analisa sua pele automaticamente.',
                image: 'camera-icon',
                action: 'next'
            },
            {
                title: 'Análise Inteligente',
                content: 'Utilizamos tecnologia LAB para encontrar a cor exata da sua pele.',
                image: 'analysis-icon',
                action: 'next'
            },
            {
                title: 'Recomendações Personalizadas',
                content: 'Receba sugestões das melhores bases O Boticário para seu tom.',
                image: 'match-icon',
                action: 'start'
            }
        ];
        
        this.createOnboardingModal(steps);
        localStorage.setItem('has_visited', 'true');
    }
    
    createOnboardingModal(steps) {
        const modal = document.createElement('div');
        modal.className = 'onboarding-modal';
        modal.innerHTML = `
            <div class="onboarding-backdrop"></div>
            <div class="onboarding-content">
                <div class="onboarding-progress">
                    ${steps.map((_, i) => `<div class="progress-dot ${i === 0 ? 'active' : ''}" data-step="${i}"></div>`).join('')}
                </div>
                <div class="onboarding-slides">
                    ${steps.map((step, i) => `
                        <div class="onboarding-slide ${i === 0 ? 'active' : ''}" data-slide="${i}">
                            <div class="slide-icon ${step.image}"></div>
                            <h2>${step.title}</h2>
                            <p>${step.content}</p>
                            <button class="btn btn-primary onboarding-btn" data-action="${step.action}">
                                ${step.action === 'start' ? 'Começar' : 'Próximo'}
                            </button>
                        </div>
                    `).join('')}
                </div>
                <button class="skip-btn">Pular</button>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add event listeners
        let currentStep = 0;
        
        modal.querySelectorAll('.onboarding-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.dataset.action === 'next') {
                    currentStep++;
                    this.updateOnboardingStep(modal, currentStep);
                } else {
                    this.closeOnboarding(modal);
                }
            });
        });
        
        modal.querySelector('.skip-btn').addEventListener('click', () => {
            this.closeOnboarding(modal);
        });
        
        // Animate in
        requestAnimationFrame(() => {
            modal.classList.add('show');
        });
    }
    
    updateOnboardingStep(modal, step) {
        // Update progress dots
        modal.querySelectorAll('.progress-dot').forEach((dot, i) => {
            dot.classList.toggle('active', i <= step);
        });
        
        // Update slides
        modal.querySelectorAll('.onboarding-slide').forEach((slide, i) => {
            slide.classList.toggle('active', i === step);
        });
        
        this.hapticFeedback('light');
    }
    
    closeOnboarding(modal) {
        modal.classList.remove('show');
        setTimeout(() => modal.remove(), 300);
        this.hapticFeedback('success');
    }
    
    /**
     * Microinteractions
     */
    setupMicrointeractions() {
        // Button press effects
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('.btn, button');
            if (btn && !btn.disabled) {
                this.createRipple(btn, e);
                this.hapticFeedback('light');
            }
        });
        
        // Input focus effects
        document.querySelectorAll('input, textarea').forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('input-focused');
            });
            
            input.addEventListener('blur', () => {
                input.parentElement.classList.remove('input-focused');
            });
        });
        
        // Hover effects with sound
        if (this.userPreferences.soundEnabled) {
            this.setupSoundEffects();
        }
    }
    
    createRipple(element, event) {
        const ripple = document.createElement('span');
        ripple.className = 'ripple-effect';
        
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        
        element.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    }
    
    /**
     * Success Celebrations
     */
    setupSuccessCelebrations() {
        // Listen for successful analysis
        window.addEventListener('analysis-complete', () => {
            this.celebrateSuccess();
        });
    }
    
    celebrateSuccess() {
        // Confetti animation
        this.createConfetti();
        
        // Success sound
        if (this.userPreferences.soundEnabled) {
            this.playSound('success');
        }
        
        // Haptic feedback
        this.hapticFeedback('success');
        
        // Show success message
        this.showSuccessMessage('Análise concluída com sucesso! 🎉');
    }
    
    createConfetti() {
        const colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8'];
        const confettiCount = 50;
        
        for (let i = 0; i < confettiCount; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.left = Math.random() * 100 + '%';
            confetti.style.animationDelay = Math.random() * 3 + 's';
            confetti.style.animationDuration = (Math.random() * 3 + 2) + 's';
            
            document.body.appendChild(confetti);
            
            setTimeout(() => confetti.remove(), 5000);
        }
    }
    
    showSuccessMessage(message) {
        const toast = document.createElement('div');
        toast.className = 'success-toast';
        toast.innerHTML = `
            <div class="toast-icon">✓</div>
            <div class="toast-message">${message}</div>
        `;
        
        document.body.appendChild(toast);
        
        requestAnimationFrame(() => {
            toast.classList.add('show');
        });
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    /**
     * Tooltips
     */
    initTooltips() {
        const tooltips = [
            { selector: '.light-meter', text: 'Indicador de qualidade da iluminação' },
            { selector: '.capture-btn', text: 'Clique para capturar foto' },
            { selector: '.match-score', text: 'Porcentagem de compatibilidade' }
        ];
        
        tooltips.forEach(({ selector, text }) => {
            document.querySelectorAll(selector).forEach(element => {
                element.setAttribute('data-tooltip', text);
                element.classList.add('has-tooltip');
            });
        });
    }
    
    /**
     * Progressive Enhancement
     */
    setupProgressiveEnhancement() {
        // Lazy load images
        this.setupLazyLoading();
        
        // Smooth scroll
        this.setupSmoothScroll();
        
        // Auto-save preferences
        this.setupAutoSave();
    }
    
    setupLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    imageObserver.unobserve(img);
                }
            });
        });
        
        images.forEach(img => imageObserver.observe(img));
    }
    
    setupSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    }
    
    setupAutoSave() {
        // Save user actions for better experience
        window.addEventListener('beforeunload', () => {
            const state = {
                lastTab: document.querySelector('.upload-tab.active')?.id,
                timestamp: Date.now()
            };
            localStorage.setItem('app_state', JSON.stringify(state));
        });
    }
    
    /**
     * Haptic Feedback
     */
    hapticFeedback(type = 'light') {
        if (!this.hapticEnabled || !this.userPreferences.hapticEnabled) return;
        
        const patterns = {
            light: [10],
            medium: [20],
            heavy: [30],
            success: [10, 50, 10],
            error: [50, 100, 50]
        };
        
        navigator.vibrate(patterns[type] || patterns.light);
    }
    
    /**
     * Sound Effects
     */
    setupSoundEffects() {
        this.sounds = {
            hover: new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiDQIG2m98OScTgwOUarm7blmFgU7k9n1unEiBC13yO/eizEIHWq+8+OWT'),
            click: new Audio('data:audio/wav;base64,UklGRqQEAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAEAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeS'),
            success: new Audio('data:audio/wav;base64,UklGRl4FAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YToFAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiDQIG2m98OScTgwOUarm7blmFgU7k9n1unEiBC13yO/eizEIHWq+8+OWTQ')
        };
        
        // Preload sounds
        Object.values(this.sounds).forEach(sound => {
            sound.volume = 0.1;
            sound.load();
        });
    }
    
    playSound(type) {
        if (this.sounds && this.sounds[type]) {
            this.sounds[type].currentTime = 0;
            this.sounds[type].play().catch(() => {});
        }
    }
    
    /**
     * User Preferences
     */
    loadPreferences() {
        const defaults = {
            soundEnabled: false,
            hapticEnabled: true,
            theme: 'light',
            language: 'pt-BR'
        };
        
        const saved = localStorage.getItem('user_preferences');
        return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
    }
    
    savePreferences() {
        localStorage.setItem('user_preferences', JSON.stringify(this.userPreferences));
    }
    
    /**
     * Social Sharing
     */
    enableSharing(results) {
        const shareData = {
            title: 'Minha Base Perfeita O Boticário',
            text: `Descobri minha base perfeita! Tom ${results.matches[0].foundation.shade} - ${results.matches[0].match_score}% de compatibilidade`,
            url: window.location.href
        };
        
        if (navigator.share) {
            // Native sharing
            return navigator.share(shareData);
        } else {
            // Fallback to copy link
            this.copyToClipboard(shareData.url);
            this.showSuccessMessage('Link copiado! 📋');
        }
    }
    
    copyToClipboard(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.uxEnhancements = new UXEnhancements();
});

// Export for use in other modules
window.UXEnhancements = UXEnhancements;
