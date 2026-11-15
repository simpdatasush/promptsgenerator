(function () {
    // Prevent duplicate initialization
    if (document.getElementById('lexi-widget-host')) return;

    // Read configuration from the script tag
    const currentScript = document.currentScript;
    const API_URL = (currentScript && currentScript.getAttribute('data-api-url')) || '/ask-lexi';
    const APP_NAME = (currentScript && currentScript.getAttribute('data-app-name')) || 'default';

    // Create host element and attach Shadow DOM for style isolation
    const host = document.createElement('div');
    host.id = 'lexi-widget-host';
    document.body.appendChild(host);
    const shadow = host.attachShadow({ mode: 'open' });

    // Inject CSS + HTML
    shadow.innerHTML = `
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&display=swap');

            :host {
                all: initial;
                font-family: 'Playfair Display', Georgia, serif;
            }

            #alex-widget-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 999999;
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                font-family: 'Playfair Display', Georgia, serif;
            }

            #alex-chat-window {
                display: none;
                width: 320px;
                height: 400px;
                max-height: 70vh;
                background: white;
                border-radius: 15px;
                box-shadow: 0 5px 25px rgba(0,0,0,0.2);
                margin-bottom: 15px;
                flex-direction: column;
                overflow: hidden;
                border: 1px solid #eee;
            }

            .lexi-header {
                background: #1a1a1a;
                color: white;
                padding: 12px 15px;
                font-size: 14px;
                font-weight: bold;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            #lexi-close {
                background: none;
                border: none;
                color: #ccc;
                font-size: 18px;
                cursor: pointer;
                line-height: 1;
                padding: 0 4px;
            }
            #lexi-close:hover { color: white; }

            #alex-messages {
                flex: 1;
                padding: 15px;
                overflow-y: auto;
                font-size: 13px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                background: #fdfdfd;
            }

            .msg-bubble {
                white-space: pre-wrap;
                word-break: break-word;
                padding: 10px 15px;
                border-radius: 15px;
                max-width: 85%;
                font-size: 13px;
                line-height: 1.4;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            .msg-user {
                align-self: flex-end;
                background: #1a1a1a;
                color: white;
                border-bottom-right-radius: 2px;
            }

            .msg-lexi {
                align-self: flex-start;
                background: #f0f0f0;
                color: #333;
                border-bottom-left-radius: 2px;
            }

            #alex-status {
                background: rgba(0,0,0,0.75);
                color: white;
                padding: 5px 12px;
                border-radius: 20px;
                margin-bottom: 8px;
                font-size: 11px;
                display: none;
                font-family: sans-serif;
            }

            .lexi-input-bar {
                display: flex;
                gap: 10px;
                background: white;
                padding: 8px 10px;
                border-radius: 30px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.12);
                border: 1px solid #eee;
                align-items: center;
            }

            #alex-input {
                border: none;
                outline: none;
                padding: 5px 12px;
                width: 220px;
                font-family: inherit;
                font-size: 14px;
                background: transparent;
                color: #1a1a1a;
            }

            #alex-send {
                background: #1a1a1a;
                color: white;
                border: none;
                border-radius: 50%;
                width: 35px;
                height: 35px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                transition: transform 0.15s ease;
            }
            #alex-send:hover { transform: scale(1.05); }
        </style>

        <div id="alex-widget-container">
            <div id="alex-chat-window">
                <div class="lexi-header">
                    <span>Lexi Concierge</span>
                    <button id="lexi-close" title="Minimize">&times;</button>
                </div>
                <div id="alex-messages"></div>
            </div>
            
            <div id="alex-status">Lexi is thinking...</div>
            
            <div class="lexi-input-bar">
                <input type="text" id="alex-input" placeholder="Ask Lexi anything..." autocomplete="off">
                <button id="alex-send" aria-label="Send message">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
        </div>
    `;

    // Query elements inside Shadow DOM
    const alexInput = shadow.getElementById('alex-input');
    const sendBtn = shadow.getElementById('alex-send');
    const closeBtn = shadow.getElementById('lexi-close');
    const alexStatus = shadow.getElementById('alex-status');
    const alexMessages = shadow.getElementById('alex-messages');
    const alexChatWindow = shadow.getElementById('alex-chat-window');

    function appendBubble(sender, text) {
        const div = document.createElement('div');
        div.className = `msg-bubble ${sender === 'user' ? 'msg-user' : 'msg-lexi'}`;
        div.innerText = text;
        alexMessages.appendChild(div);
        alexMessages.scrollTop = alexMessages.scrollHeight;
        return div;
    }

    function typeWriter(element, text, index) {
        if (index < text.length) {
            element.innerText += text.charAt(index);
            alexMessages.scrollTop = alexMessages.scrollHeight;
            setTimeout(() => typeWriter(element, text, index + 1), 25);
        }
    }

    async function handleAlexChat() {
        const text = alexInput.value.trim();
        if (!text) return;

        alexChatWindow.style.display = 'flex';
        appendBubble('user', text);
        alexInput.value = '';
        alexStatus.style.display = 'block';
        alexStatus.innerText = "Lexi is thinking...";

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message: text,
                    source_app: APP_NAME,
                    page_url: window.location.href
                })
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();

            alexStatus.style.display = 'none';
            const lexiBubble = appendBubble('lexi', "");
            typeWriter(lexiBubble, data.reply || "No response received.", 0);
        } catch (err) {
            console.error("Lexi Widget Error:", err);
            alexStatus.innerText = "Service unavailable.";
        }
    }

    // Event Listeners
    sendBtn.addEventListener('click', (e) => {
        e.preventDefault();
        handleAlexChat();
    });

    alexInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleAlexChat();
        }
    });

    // Re-open window if user focuses input after closing
    alexInput.addEventListener('focus', () => {
        if (alexMessages.children.length > 0) {
            alexChatWindow.style.display = 'flex';
        }
    });

    closeBtn.addEventListener('click', () => {
        alexChatWindow.style.display = 'none';
    });
})();
