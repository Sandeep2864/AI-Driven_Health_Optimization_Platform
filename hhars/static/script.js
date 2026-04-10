// static/script.js
document.addEventListener('DOMContentLoaded', () => {
  const allergyInput = document.getElementById('allergyInput');
  const allergyTags = document.getElementById('allergyTags');
  const allergiesHidden = document.getElementById('allergiesHidden');

  function updateHidden() {
    const values = Array.from(allergyTags.querySelectorAll('.tag')).map(t => t.dataset.value);
    allergiesHidden.value = values.join(',');
  }

  function addTag(val) {
    val = (val||'').trim();
    if (!val) return;
    const exists = Array.from(allergyTags.querySelectorAll('.tag')).some(t => t.dataset.value.toLowerCase() === val.toLowerCase());
    if (exists) { allergyInput.value = ''; return; }
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.dataset.value = val;
    tag.innerHTML = `<span>${val}</span><button type="button" class="remove">×</button>`;
    tag.querySelector('.remove').addEventListener('click', () => { tag.remove(); updateHidden(); });
    allergyTags.appendChild(tag);
    allergyInput.value = '';
    updateHidden();
  }

  allergyInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTag(allergyInput.value);
    }
  });

  // support clicking datalist option by pressing Tab
  allergyInput.addEventListener('blur', () => {
    if (allergyInput.value.trim()) addTag(allergyInput.value.trim());
  });

  // main actions
  document.getElementById('getBtn').addEventListener('click', runRecommendation);
  document.getElementById('clearBtn').addEventListener('click', () => {
    document.getElementById('userForm').reset();
    allergyTags.innerHTML = '';
    allergiesHidden.value = '';
    document.getElementById('resultsPanel').classList.add('hidden');
    const chat = document.getElementById('chatArea'); chat.innerHTML = `<div class="bot-message"><div class="avatar">🤖</div><div class="bubble"><strong>Hi — I'm HHARS.</strong> Enter your details and press Get Recommendation.</div></div>`;
  });

  async function runRecommendation() {
    const form = document.getElementById('userForm');
    const fd = new FormData(form);
    // ensure allergies hidden is present
    updateHidden();
    const payload = {};
    fd.forEach((v,k) => payload[k] = v);
    try {
      const r1 = await fetch('/api/predict', { method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if (!r1.ok) { alert('Predict failed'); return; }
      const pred = await r1.json();
      // show a user message
      const chat = document.getElementById('chatArea');
      const userMsg = document.createElement('div'); userMsg.className='user-message'; userMsg.innerHTML = `<div class="bubble">Generating plan…</div>`;
      chat.appendChild(userMsg);
      chat.scrollTop = chat.scrollHeight;

      // display metrics
      const rec = pred.recommended || {};
      document.getElementById('calories').innerText = Math.round(rec.Recommended_Calories || rec.RecommendedCalories || 0);
      document.getElementById('protein').innerText = Math.round(rec.Recommended_Protein || 0);
      document.getElementById('carbs').innerText = Math.round(rec.Recommended_Carbs || 0);
      document.getElementById('fats').innerText = Math.round(rec.Recommended_Fats || 0);

      const risk = pred.risk_label || 'Unknown';
      const badge = document.getElementById('riskBadge');
      badge.innerText = 'Risk: ' + risk;
      badge.className = 'risk ' + (risk.toLowerCase() === 'high' ? 'high' : (risk.toLowerCase() === 'moderate' ? 'medium' : 'low'));

      // call recommend
      payload.recommended = pred.recommended;
      const r2 = await fetch('/api/recommend', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      if (!r2.ok) { alert('Recommendation failed'); return; }
      const plan = await r2.json();

      // render week
// Replace the (plan.week_plan || []).forEach section with this:
(plan.week_plan || []).forEach((dayMeals, index) => {
  const card = document.createElement('div'); 
  card.className = 'day-card';
  
  const title = document.createElement('div'); 
  title.className = 'day-title'; 
  title.innerText = 'Day ' + (index + 1); // Use the index since 'day.day' doesn't exist
  
  card.appendChild(title);
  const list = document.createElement('div'); 
  list.className = 'meal-list';
  
  dayMeals.forEach(m => {
    const item = document.createElement('div'); 
    item.className = 'meal-item';
    item.innerHTML = `<div class="meal-name">${m.name}</div><div class="meal-meta">${m.calories} kcal</div>`;
    list.appendChild(item);
  });
  
  // Calculate total calories for the day manually or pass it from backend
  const dailyTotal = dayMeals.reduce((sum, m) => sum + m.calories, 0);
  
  const footer = document.createElement('div'); 
  footer.className = 'day-footer';
  footer.innerText = 'Total: ' + Math.round(dailyTotal) + ' kcal';
  
  card.appendChild(list); 
  card.appendChild(footer);
  weekGrid.appendChild(card);
});

      
      document.getElementById('resultsPanel').classList.remove('hidden');

      const bot = document.createElement('div'); bot.className='bot-message';
      bot.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><strong>Done:</strong> Generated a personalized 7‑day meal plan. You can adjust preferences and regenerate anytime.</div>`;
      chat.appendChild(bot);
      chat.scrollTop = chat.scrollHeight;

    } catch (err) {
      console.error(err);
      alert('Error: ' + err.message);
    }
  }

});
