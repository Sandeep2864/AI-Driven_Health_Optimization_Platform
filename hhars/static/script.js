// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const userForm = document.getElementById('userForm');
    const allergyInput = document.getElementById('allergyInput');
    const allergyTags = document.getElementById('allergyTags');
    const allergiesHidden = document.getElementById('allergiesHidden');
    const chatArea = document.getElementById('chatArea');
    const resultsPanel = document.getElementById('resultsPanel');
    const weekGrid = document.getElementById('weekGrid');
    const dishesGrid = document.getElementById('dishesGrid');
    const nearbySection = document.getElementById('nearbyDishes');
    const getBtn = document.getElementById('getBtn');
    const resultsToggleBtn = document.getElementById('resultsToggleBtn');

    /**
     * ALLERGY TAG LOGIC - Manages visual tags and hidden input
     */
    function updateHiddenAllergies() {
        const values = Array.from(allergyTags.querySelectorAll('.tag')).map(t => t.dataset.value);
        allergiesHidden.value = values.join(',');
    }

    function addAllergyTag(val) {
        val = (val || '').trim();
        if (!val) return;
        
        const exists = Array.from(allergyTags.querySelectorAll('.tag'))
            .some(t => t.dataset.value.toLowerCase() === val.toLowerCase());
        
        if (exists) { 
            allergyInput.value = ''; 
            return; 
        }

        const tag = document.createElement('div');
        tag.className = 'tag animate__animated animate__fadeIn';
        tag.dataset.value = val;
        tag.innerHTML = `<span>${val}</span><button type="button" class="remove">×</button>`;
        
        tag.querySelector('.remove').addEventListener('click', () => { 
            tag.remove(); 
            updateHiddenAllergies(); 
        });

        allergyTags.appendChild(tag);
        allergyInput.value = '';
        updateHiddenAllergies();
    }

    allergyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            addAllergyTag(allergyInput.value);
        }
    });

    /**
     * CHAT UI HELPERS
     */
    function addChatMessage(role, content, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `${role}-message animate__animated animate__fadeIn`;
        
        if (role === 'bot') {
            msgDiv.innerHTML = `
                <div class="avatar">🤖</div>
                <div class="bubble">${isHtml ? content : `<p>${content}</p>`}</div>
            `;
        } else {
            msgDiv.innerHTML = `<div class="bubble">${content}</div>`;
        }
        
        chatArea.appendChild(msgDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    /**
     * MAIN RECOMMENDATION ENGINE
     */
    async function runRecommendation() {
        // 1. Prepare Data
        updateHiddenAllergies();
        const formData = new FormData(userForm);
        const data = Object.fromEntries(formData.entries());
        
        if (window.userLocation) {
            data.location = JSON.stringify(window.userLocation);
        }

        // 2. UI Loading State
        getBtn.disabled = true;
        const btnText = getBtn.querySelector('.btn-text');
        const btnLoader = getBtn.querySelector('.btn-loader');
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');

        addChatMessage('user', 'Analyzing my health profile and generating a plan...');

        try {
            // STEP 1: Predict health metrics (Backend: /api/predict)
            const r1 = await fetch('/api/predict', { 
                method: 'POST', 
                headers: {'Content-Type': 'application/json'}, 
                body: JSON.stringify(data) 
            });
            
            if (!r1.ok) throw new Error('Health prediction failed.');
            const pred = await r1.json();

            // Display Predicted Metrics in the Sidebar/Panel
            const rec = pred.recommended || {};
            document.getElementById('calories').innerText = Math.round(rec.Recommended_Calories || 0);
            document.getElementById('protein').innerText = Math.round(rec.Recommended_Protein || 0);
            document.getElementById('carbs').innerText = Math.round(rec.Recommended_Carbs || 0);
            document.getElementById('fats').innerText = Math.round(rec.Recommended_Fats || 0);

            // Update Health Risk Badge
            const risk = pred.health_label || 'Unknown';
            const badge = document.getElementById('riskBadge');
            badge.innerText = `Status: ${risk}`;
            badge.className = 'risk ' + (risk.toLowerCase().includes('poor') ? 'high' : 
                                       (risk.toLowerCase().includes('average') ? 'medium' : 'low'));

            // STEP 2: Get Meal Plan (Backend: /api/recommend)
            // We merge the original user data with the prediction results
            const recommendationPayload = {
                ...data,
                recommended: rec 
            };

            const r2 = await fetch('/api/recommend', { 
                method: 'POST', 
                headers: {'Content-Type': 'application/json'}, 
                body: JSON.stringify(recommendationPayload) 
            });

            if (!r2.ok) throw new Error('Meal recommendation failed.');
            const plan = await r2.json();

            // STEP 3: Render Nearby Dishes (Local features)
            dishesGrid.innerHTML = '';
            if (plan.nearby_dishes && plan.nearby_dishes.length > 0) {
                plan.nearby_dishes.forEach(dish => {
                    const dCard = document.createElement('div');
                    dCard.className = 'dish-card';
                    dCard.innerHTML = `
                        <div class="dish-name">${dish['Food Name']}</div>
                        <div class="dish-meta">
                            <span>${dish.Type}</span> • <span>${dish.State}</span>
                        </div>
                    `;
                    dishesGrid.appendChild(dCard);
                });
                nearbySection.classList.remove('hidden');
            }

            // STEP 4: Render Weekly Plan Grid
            weekGrid.innerHTML = '';
            const dayNames = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
            
            (plan.week_plan || []).forEach((dayMeals, index) => {
                const card = document.createElement('div');
                card.className = 'day-card animate__animated animate__fadeInUp';
                card.style.animationDelay = `${index * 0.1}s`;

                const title = document.createElement('div');
                title.className = 'day-title';
                title.innerText = dayNames[index] || `Day ${index + 1}`;

                const list = document.createElement('div');
                list.className = 'meal-list';

                let dayCalories = 0;
                dayMeals.forEach(m => {
                    dayCalories += m.calories || 0;
                    const item = document.createElement('div');
                    item.className = 'meal-item';
                    item.innerHTML = `
                        <div class="meal-name">${m.name}</div>
                        <div class="meal-meta">${Math.round(m.calories)} kcal | P: ${Math.round(m.protein)}g</div>
                    `;
                    list.appendChild(item);
                });

                const footer = document.createElement('div');
                footer.className = 'day-footer';
                footer.innerText = `Daily Total: ${Math.round(dayCalories)} kcal`;

                card.appendChild(title);
                card.appendChild(list);
                card.appendChild(footer);
                weekGrid.appendChild(card);
            });

            // Reveal the Results UI
            resultsPanel.classList.remove('hidden');
            resultsToggleBtn.style.display = 'flex';
            
            addChatMessage('bot', `<strong>Plan Ready!</strong> I've optimized your meals for <strong>${Math.round(rec.Recommended_Calories)} kcal</strong>.`, true);

        } catch (err) {
            console.error("Workflow Error:", err);
            addChatMessage('bot', `⚠️ Error: ${err.message}`);
        } finally {
            getBtn.disabled = false;
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
        }
    }

    /**
     * INITIALIZATION & EVENT LISTENERS
     */
    getBtn.addEventListener('click', runRecommendation);

    document.getElementById('clearBtn').addEventListener('click', () => {
        userForm.reset();
        allergyTags.innerHTML = '';
        allergiesHidden.value = '';
        resultsPanel.classList.add('hidden');
        resultsToggleBtn.style.display = 'none';
        chatArea.innerHTML = '';
        addChatMessage('bot', '<strong>Hi — I\'m HHARS.</strong> Everything has been reset. Ready for a new plan?', true);
    });

    // Handle results panel visibility
    resultsToggleBtn.addEventListener('click', () => {
        const isCollapsed = resultsToggleBtn.classList.contains('collapsed');
        if (isCollapsed) {
            resultsToggleBtn.classList.replace('collapsed', 'expanded');
            resultsPanel.classList.remove('hidden');
        } else {
            resultsToggleBtn.classList.replace('expanded', 'collapsed');
            resultsPanel.classList.add('hidden');
        }
    });
});
