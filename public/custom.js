// Custom JS to handle UI enhancements, branding, and transcription injection

console.log("Custom JS loaded");

const TRANSCRIPTION_PREFIX = "///TRANSCRIPTION///";

// ─── GLOBAL ANIMATION STATE ───────────────────────────
let animationActive = true;

// ─── AVATAR SWAP ─────────────────────────────────────
const NVIDIA_AVATAR_SRC = '/public/nvidia-mark.svg';

function swapAvatar() {
    const avatars = document.querySelectorAll(
        'img[alt^="Avatar for"], img[alt="Assistant Avatar"], img[alt="Avatar"]'
    );
    avatars.forEach(img => {
        if (img.getAttribute('src') !== NVIDIA_AVATAR_SRC) {
            img.src = NVIDIA_AVATAR_SRC;
        }
    });
}

// ─── VOICE BUTTON STYLING ─────────────────────────────
const styleVoiceButton = () => {
    const buttons = document.querySelectorAll('#message-composer button');
    buttons.forEach(btn => {
        const svg = btn.querySelector('svg');
        if (svg && btn.id !== 'chat-submit') {
            btn.id = 'voice-button';
            btn.addEventListener('click', () => {
                animationActive = false;
                const welcomeDiv = document.querySelector('#welcome-text');
                if (welcomeDiv) {
                    welcomeDiv.style.opacity = '0';
                    setTimeout(() => welcomeDiv.remove(), 500);
                }
            });
        }
    });
};

const syncThinkingIndicator = () => {
    const isThinking = Boolean(document.getElementById('stop-button'));
    document.documentElement.classList.toggle('nvidia-thinking', isThinking);

    const currentAvatar = document.querySelector('.nvidia-thinking-avatar');
    const avatarImages = document.querySelectorAll(
        'img[alt^="Avatar for"], img[alt="Assistant Avatar"], img[alt="Avatar"]'
    );
    const latestImage = isThinking ? avatarImages[avatarImages.length - 1] : null;
    const nextAvatar = latestImage
        ? latestImage.closest(
            'span.relative.flex.shrink-0.overflow-hidden.rounded-full'
        ) || latestImage.parentElement
        : null;

    if (currentAvatar && currentAvatar !== nextAvatar) {
        currentAvatar.classList.remove('nvidia-thinking-avatar');
    }
    if (nextAvatar) {
        nextAvatar.classList.add('nvidia-thinking-avatar');
    }
};

let retryCount = 0;
const retryInterval = setInterval(() => {
    swapAvatar();
    retryCount++;
    if (retryCount >= 30) clearInterval(retryInterval);
}, 1000);

const themeObserver = new MutationObserver(swapAvatar);
themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
});

swapAvatar();

// ─── TRANSCRIPTION INJECTION ──────────────────────────
function setNativeValue(element, value) {
    const valueSetter = Object.getOwnPropertyDescriptor(element, 'value').set;
    const prototype = Object.getPrototypeOf(element);
    const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value').set;
    if (valueSetter && valueSetter !== prototypeValueSetter) {
        prototypeValueSetter.call(element, value);
    } else {
        valueSetter.call(element, value);
    }
    element.dispatchEvent(new Event('input', { bubbles: true }));
}

const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
            if (node.nodeType === 1) {
                const contentElements = node.querySelectorAll('.step-content, .message-content');
                const elementsToCheck = Array.from(contentElements);
                if (node.classList.contains('step-content') || node.classList.contains('message-content')) {
                    elementsToCheck.push(node);
                }
                elementsToCheck.forEach((element) => {
                    if (element.textContent && element.textContent.includes(TRANSCRIPTION_PREFIX)) {
                        handleTranscriptionNode(element);
                    }
                });
            }
        });
    });
    syncThinkingIndicator();
    setTimeout(swapAvatar, 500);
    setTimeout(styleVoiceButton, 500);
});

function handleTranscriptionNode(node) {
    const text = node.textContent;
    if (!text.includes(TRANSCRIPTION_PREFIX)) return;
    const cleanText = text.split(TRANSCRIPTION_PREFIX)[1].trim();
    console.log("Found transcription:", cleanText);

    animationActive = false;
    const welcomeDiv = document.querySelector('#welcome-text');
    if (welcomeDiv) {
        welcomeDiv.style.opacity = '0';
        setTimeout(() => welcomeDiv.remove(), 500);
    }

    const input = document.getElementById('chat-input');
    if (input) {
        setNativeValue(input, cleanText);
        input.focus();
    } else {
        console.warn("Chat input not found");
    }
    const stepContainer = node.closest('.step');
    const messageContainer = node.closest('.message');
    if (stepContainer) {
        stepContainer.style.display = 'none';
    } else if (messageContainer) {
        messageContainer.style.display = 'none';
    } else {
        node.style.display = 'none';
    }
}

// ─── BRANDING ─────────────────────────────────────────
const addBranding = () => {
    if (document.querySelector('#app-brand')) return;
    const brand = document.createElement('div');
    brand.id = 'app-brand';
    // Sanitized: Generic app title
    brand.innerText = 'RAG Assistant';
    brand.style.cssText = `
        position: fixed;
        top: 14px;
        left: 60px;
        font-size: 18px;
        font-weight: 600;
        z-index: 9999;
        font-family: sans-serif;
        color: ${document.documentElement.classList.contains('dark') ? 'white' : 'black'};
    `;
    document.body.appendChild(brand);
};

// ─── BRANDING THEME OBSERVER ──────────────────────────
const brandingThemeObserver = new MutationObserver(() => {
    const brand = document.querySelector('#app-brand');
    if (brand) {
        brand.style.color = document.documentElement.classList.contains('dark') ? 'white' : 'black';
    }
    const welcome = document.querySelector('#welcome-text');
    if (welcome) {
        welcome.style.color = document.documentElement.classList.contains('dark') ? 'white' : 'black';
    }
});

brandingThemeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class']
});

setTimeout(addBranding, 500);
setTimeout(styleVoiceButton, 500);

// ─── WELCOME TEXT ANIMATION ───────────────────────────
const addWelcomeText = () => {
    if (document.querySelector('#welcome-text')) return;

    const welcomeDiv = document.createElement('div');
    welcomeDiv.id = 'welcome-text';
    welcomeDiv.style.cssText = `
        position: relative;
        font-size: 22px;
        font-weight: 500;
        font-family: sans-serif;
        text-align: center;
        width: min(90vw, 720px);
        order: 2;
        pointer-events: none;
        transition: opacity 0.5s ease;
        color: ${document.documentElement.classList.contains('dark') ? 'white' : 'black'};
    `;
    const welcomeScreen = document.getElementById('welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.appendChild(welcomeDiv);
    } else {
        document.body.appendChild(welcomeDiv);
    }

    const messages = [
        'Ask me anything about NVIDIA financial reports...',
    ];

    animationActive = true;
    let msgIndex = 0;

    const showDots = (callback) => {
        welcomeDiv.style.direction = 'ltr';
        welcomeDiv.innerText = '';
        let dots = 0;
        const interval = setInterval(() => {
            if (!animationActive) { clearInterval(interval); return; }
            dots++;
            welcomeDiv.innerText = '.'.repeat(dots);
            if (dots >= 3) {
                clearInterval(interval);
                setTimeout(() => {
                    welcomeDiv.innerText = '';
                    if (callback) callback();
                }, 400);
            }
        }, 300);
    };

    const typeWriter = (text, callback) => {
        welcomeDiv.innerText = '';
        welcomeDiv.style.direction = 'ltr';
        let i = 0;
        const interval = setInterval(() => {
            if (!animationActive) { clearInterval(interval); return; }
            i++;
            welcomeDiv.innerText = text.substring(0, i);
            if (i >= text.length) {
                clearInterval(interval);
                if (callback) setTimeout(callback, 2000);
            }
        }, 80);
    };

    const eraseText = (callback) => {
        let text = welcomeDiv.innerText;
        const interval = setInterval(() => {
            if (!animationActive) { clearInterval(interval); return; }
            text = text.slice(0, -1);
            welcomeDiv.innerText = text;
            if (text.length === 0) {
                clearInterval(interval);
                if (callback) setTimeout(callback, 300);
            }
        }, 50);
    };

    const animate = () => {
        if (!animationActive) return;
        showDots(() => {
            typeWriter(messages[msgIndex], () => {
                eraseText(() => {
                    msgIndex = (msgIndex + 1) % messages.length;
                    animate();
                });
            });
        });
    };

    animate();

    const attachInputListener = () => {
        const input = document.getElementById('chat-input');
        if (input) {
            input.addEventListener('input', () => {
                if (input.value.length > 0) {
                    animationActive = false;
                    welcomeDiv.style.opacity = '0';
                    setTimeout(() => welcomeDiv.remove(), 500);
                }
            });
        } else {
            setTimeout(attachInputListener, 500);
        }
    };

    attachInputListener();
};

setTimeout(addWelcomeText, 1000);

// ─── RE-RUN WELCOME ON NEW CHAT ───────────────────────
const watchNewChatButton = () => {
    const newChatBtn = document.querySelector('#new-chat-button');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', () => {
            const existing = document.querySelector('#welcome-text');
            if (existing) existing.remove();
            setTimeout(addWelcomeText, 1000);
        });
    } else {
        setTimeout(watchNewChatButton, 500);
    }
};

watchNewChatButton();

const targetNode = document.body;
const config = { childList: true, subtree: true, characterData: true };
observer.observe(targetNode, config);
syncThinkingIndicator();