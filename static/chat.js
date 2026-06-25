let current_huff_result = null; // The most recent Huff tool result returned by the server. Used to avoid re-rendering if the model returns the same result twice in a row.
let result_array = [];          // Array of Huff tool results for comparison. Each element is a Huff result object.
let compareChartInst = null;    // Chart.js instance for the comparison chart. We destroy and recreate it when the data changes.

const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

// Declared at top level so we can destroy the old chart instance before redrawing.
let competitorsChart = null;
// Separate Chart instance for the market-share pie
let marketShareChart = null;
// Your store's predicted visits/day, drawn as a reference line on the bar chart.
let candidateRefVisits = 0;

// Canonical conversation history (server-authoritative). Holds user / assistant /
// tool messages in OpenAI format. The system prompt lives on the server, so it is
// NOT stored here. After each turn we replace this with what the server returns.
let conversation = [];
let selectedLocation = null;
let busy = false;

addBotMessage(
  "Welcome. I'll help you evaluate a store location in Worcester, MA using a Huff " +
  "gravity model. Just tell me, in any order: the business (a NAICS code like 4441), " +
  "where you're considering (click the map or type coordinates such as 42.24, -71.78), " +
  "and the proposed floor area in square meters."
);

sendBtn.addEventListener("click", handleSend);

chatInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    handleSend();
  }
});

// Called by map.js when the user clicks the map. We record the location and let the
// assistant acknowledge it, so the model stays the single source of conversation control.
window.onMapLocationSelected = function (location) {
  selectedLocation = { lat: location.lat, lon: location.lon };
  sendUserTurn(
    `I selected a candidate location on the map: ${location.lat.toFixed(6)}, ${location.lon.toFixed(6)}.`
  );
};

function handleSend() {
  const text = chatInput.value.trim();
  if (!text || busy) return;
  chatInput.value = "";
  sendUserTurn(text);
}

async function sendUserTurn(text) {
  if (busy) return;
  busy = true;
  sendBtn.disabled = true;

  addUserMessage(text);
  conversation.push({ role: "user", content: text });

  const typing = addBotMessage("…");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: conversation,
        selected_location: selectedLocation
      })
    });

    const data = await response.json();
    removeMessage(typing);

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "The assistant could not respond.");
    }

    // The server returns the authoritative history, including any tool calls/results.
    if (Array.isArray(data.messages)) {
      conversation = data.messages;
    }

    addBotMessage(data.reply || "(no reply)");

    // If the model ran the Huff tool this turn, render its real output.
    if (data.huff_result) {
      current_huff_result = data.huff_result;
      const r = data.huff_result;
      renderResult(r);
      updateCompetitorsChart(r.competitors, r.predicted_visits);
      renderMarketSharePie(r);

      if (window.setCandidateLocation &&
          typeof r.candidate_lat === "number" &&
          typeof r.candidate_lon === "number") {
        window.setCandidateLocation(r.candidate_lat, r.candidate_lon, false);
      }

      if (window.plotCompetitors && Array.isArray(r.competitors)) {
        window.plotCompetitors(r.competitors);
      }

      // ==========================================
      // Switch to the results view (show the chart and table, hide the map) so the user sees the output immediately.
      // ==========================================
      const showResultBtn = document.getElementById("showResultBtn");
      const showMapBtn = document.getElementById("showMapBtn");
      const chartSection = document.getElementById("chartSection");
      const modelResultPanel = document.getElementById("modelResultPanel");
      const mapPanel = document.querySelector(".map-panel");

      // 1. Update the button states
      if (showResultBtn) showResultBtn.classList.add("active");
      if (showMapBtn) showMapBtn.classList.remove("active");

      // 2. Show the results panel
      if (chartSection) chartSection.classList.remove("hidden");
      if (modelResultPanel) modelResultPanel.classList.remove("hidden");
      if (mapPanel) mapPanel.classList.add("hidden");

      // 3. Scroll to the "tabs" element with a smooth deceleration effect
      const toolElement = document.getElementById("tabs");
      if (toolElement) {
        // Calculate the target scroll position (top of the tabs element relative to the document)
        const targetPosition = toolElement.getBoundingClientRect().top + window.scrollY;
        
        // Define a function that performs the smooth scrolling with deceleration
        const easeOutScroll = () => {
          const currentPosition = window.scrollY;
          // Calculate the distance to the target position
          const distance = targetPosition - currentPosition;
          
          // If the distance is small enough, scroll directly to the position and stop the animation
          if (Math.abs(distance) < 1) {
            window.scrollTo(0, targetPosition);
            return;
          }
          
          // Core deceleration formula: move 12% of the remaining distance each frame
          // Large remaining distances result in faster movement, small remaining distances result in slower movement
          window.scrollTo(0, currentPosition + distance * 0.05);
          requestAnimationFrame(easeOutScroll);
        };
        
        requestAnimationFrame(easeOutScroll);
      }
    }
  } catch (error) {
    removeMessage(typing);
    addErrorMessage(error.message || String(error));
  } finally {
    busy = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

function renderResult(result) {
  const summary = document.getElementById("resultSummary");
  const tableWrap = document.getElementById("competitorTable");

  const visits = result.predicted_visits;
  const visitsText = Number.isFinite(Number(visits)) ? Number(visits).toFixed(2) : "N/A";
  const marketShare = Number(result.market_share);
  const sharePct = Number.isFinite(marketShare) ? (marketShare * 100).toFixed(4) : null;
  const totalDemand = result.total_demand;
  const runtime = result.runtime_ms ?? "N/A";
  const floorArea = result.floor_area;
  const notes = result.notes ?? "";

  summary.innerHTML = `
    <div class="scorecard">
      <div class="card headline">
        <div class="card-value">${escapeHtml(visitsText)}<span class="card-unit">visits/day</span></div>
        <div class="card-label">Predicted visits <span class="card-tag">key result</span></div>
      </div>
      <div class="card">
        <div class="card-value">${sharePct !== null ? sharePct : "N/A"}<span class="card-unit">%</span></div>
        <div class="card-label">Market share</div>
      </div>
      <div class="card">
        <div class="card-value">${totalDemand != null ? escapeHtml(totalDemand) : "N/A"}<span class="card-unit">visits/day</span></div>
        <div class="card-label">Worcester demand</div>
      </div>
      <div class="prediction-highlight">
  Main prediction: this location is expected to receive ${escapeHtml(visitsText)} visits/day.
</div>
    </div>
    <div class="result-meta">
      ${floorArea ? `<span><strong>Floor area:</strong> ${escapeHtml(floorArea)} m²</span>` : ""}
      <span><strong>Runtime:</strong> ${escapeHtml(runtime)} ms</span>
    </div>
    ${notes ? `<div class="result-note">${escapeHtml(notes)}</div>` : ""}
  `;

  const stores = Array.isArray(result.competitors) ? result.competitors : [];

  if (stores.length === 0) {
    tableWrap.innerHTML = "No competitor businesses found for this category.";
    return;
  }

  tableWrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Store</th>
          <th>Distance (mi)</th>
          <th>Size (m²)</th>
          <th>Visits/day</th>
          <th>Huff pull (rel.)</th>
        </tr>
      </thead>
      <tbody>
        ${stores.map(c => `
          <tr>
            <td>${escapeHtml(c.name ?? "Unknown")}</td>
            <td>${escapeHtml(c.distance_miles ?? "N/A")}</td>
            <td>${escapeHtml(c.size ?? "N/A")}</td>
            <td>${escapeHtml(c.visits_per_day ?? "N/A")}</td>
            <td>${escapeHtml(c.huff_pull ?? c.attraction ?? "N/A")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderMarketSharePie(result) {
  const pieCanvas = document.getElementById('marketSharePie');
  if (!pieCanvas) return;

  // Share of daily visits among your store (predicted) and the nearby competitors.
  // This is the split AMONG these players — not your share of all Worcester demand
  // (that true figure, ~0.45%, stays in the scorecard). No "Other" catch-all slice.
  const competitors = Array.isArray(result.competitors) ? result.competitors : [];
  const candidateVisits = Number(result.predicted_visits) || 0;

  const labels = ['Your store'];
  const values = [candidateVisits];
  competitors.slice(0, 10).forEach(c => {
    const v = Number(c.visits_per_day) || 0;
    if (v > 0) {
      labels.push(c.name ?? 'Competitor');
      values.push(v);
    }
  });

  const total = values.reduce((a, b) => a + b, 0) || 1;

  const palette = [
    'rgba(14, 165, 164, 0.9)',   // Your store — teal highlight
    'rgba(99, 102, 241, 0.7)',
    'rgba(236, 72, 153, 0.7)',
    'rgba(34, 197, 94, 0.7)',
    'rgba(249, 115, 22, 0.7)',
    'rgba(59, 130, 246, 0.7)',
    'rgba(168, 85, 247, 0.7)',
    'rgba(20, 184, 166, 0.7)',
    'rgba(245, 158, 11, 0.7)',
    'rgba(100, 116, 139, 0.7)',
    'rgba(190, 24, 93, 0.6)'
  ];

  if (marketShareChart) {
    try { marketShareChart.destroy(); } catch (e) {}
    marketShareChart = null;
  }

  const ctx = pieCanvas.getContext('2d');
  marketShareChart = new Chart(ctx, {
    type: 'pie',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: labels.map((_, i) => palette[i % palette.length]),
        borderColor: 'rgba(255,255,255,0.9)',
        borderWidth: 1
      }]
    },
    options: {
      plugins: {
        legend: { position: 'right', labels: { boxWidth: 12, padding: 6 } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const v = Number(ctx.raw) || 0;
              const pct = (v / total) * 100;
              return `${ctx.label}: ${pct.toFixed(1)}% (${v.toFixed(1)} visits/day)`;
            }
          }
        }
      },
      responsive: true,
      maintainAspectRatio: false
    }
  });
}

function updateCompetitorsChart(competitors, candidatePredVisits) {
  const placeholder = document.getElementById('chartPlaceholder1');
  const topCompetitorsChartCanvas = document.getElementById('topCompetitorsChart');
  const marketSharePieCanvas = document.getElementById('marketSharePie');
  const section = document.getElementById("chartSection");
  if (!section) return;

  // Hide and tear down the chart if there is no data.
  if (!competitors || competitors.length === 0) {
    section.classList.add("hidden");
    placeholder.style.display = 'flex';
    topCompetitorsChartCanvas.style.zIndex = '0';
    marketSharePieCanvas.style.zIndex = '0';
    if (competitorsChart) {
      competitorsChart.destroy();
      competitorsChart = null;
    }
    return;
  }

  section.classList.remove("hidden");
  placeholder.style.display = 'none';
  topCompetitorsChartCanvas.style.zIndex = '2';
  marketSharePieCanvas.style.zIndex = '2';

  // Busiest competitor stores among the nearby set (by real historical visits/day).
  const top10 = [...competitors]
    .sort((a, b) => (Number(b.visits_per_day) || 0) - (Number(a.visits_per_day) || 0))
    .slice(0, 10);

  const labels = top10.map(c => c.name ?? "Unknown");
  const data = top10.map(c => Number(c.visits_per_day) || 0);

  // Reference line = your store's predicted visits/day. Stored at module scope so
  // the plugin redraws it correctly on in-place (animated) updates too.
  candidateRefVisits = (typeof candidatePredVisits === 'number' && !isNaN(candidatePredVisits))
    ? Number(candidatePredVisits)
    : 0;

  // Update in place so bars animate smoothly when the numbers change.
  if (competitorsChart) {
    competitorsChart.data.labels = labels;
    competitorsChart.data.datasets[0].data = data;
    competitorsChart.update();
    return;
  }

  const ctx = document.getElementById("topCompetitorsChart").getContext("2d");

  // Draws a dashed line at your store's predicted visits/day, for comparison.
  const drawCandidateLinePlugin = {
    id: 'drawCandidateLine',
    afterDatasetsDraw: (chart) => {
      const yScale = chart.scales['y'];
      if (!yScale) return;
      const yPixel = yScale.getPixelForValue(candidateRefVisits);

      const c2 = chart.ctx;
      c2.save();
      c2.beginPath();
      c2.moveTo(chart.chartArea.left, yPixel);
      c2.lineTo(chart.chartArea.right, yPixel);
      c2.lineWidth = 2;
      c2.strokeStyle = 'rgba(245, 158, 11, 0.95)';
      c2.setLineDash([6, 4]);
      c2.stroke();

      c2.fillStyle = 'rgba(180, 83, 9, 0.95)';
      c2.font = '12px Arial';
  const label = `Your store (predicted): ${candidateRefVisits.toFixed(2)}/day`;
      const tw = c2.measureText(label).width;
      c2.fillText(label, chart.chartArea.right - tw - 6, yPixel - 6);
      c2.restore();
    }
  };

  competitorsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Visits/day",
        data: data,
        backgroundColor: "rgba(14, 165, 164, 0.55)",
        hoverBackgroundColor: "rgba(14, 165, 164, 0.85)",
        borderColor: "rgba(12, 128, 127, 1)",
        borderWidth: 1,
        borderRadius: 6,
        maxBarThickness: 64
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 700, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15, 33, 56, 0.92)",
          padding: 10,
          callbacks: {
            label: (item) => `Visits/day: ${Number(item.parsed.y).toFixed(1)}`
          }
        }
      },
         scales: {
        y: {
          type: "logarithmic",
          title: {
            display: true,
            text: "Visits/day (log scale)"
          },
          grid: {
            color: "rgba(15, 23, 42, 0.06)"
          }
        },
        x: {
          grid: { display: false },
          ticks: { maxRotation: 45, minRotation: 30, autoSkip: false, font: { size: 10 } }
        }
      }
    },
    plugins: [drawCandidateLinePlugin]
  });
}

function addBotMessage(text) {
  return addMessage(text, "bot");
}

function addUserMessage(text) {
  return addMessage(text, "user");
}

function addErrorMessage(text) {
  return addMessage(text, "error");
}

function addMessage(text, type) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.innerText = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function removeMessage(el) {
  if (el && el.parentNode) {
    el.parentNode.removeChild(el);
  }
}


function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ==========================================
// New features: Save, Export, Import, and Compare panel logic
// ==========================================

// 1. Tab switching logic (Modified for forced display override)
const tabCurrent = document.getElementById('tabCurrent');
const tabSaved = document.getElementById('tabSaved');
const toolPanel = document.getElementById('tool');
const savedPanel = document.getElementById('savedPanel');

// Force the initial state: show Current panel, hide Saved panel
if (savedPanel) {
  savedPanel.style.display = 'none';
}

if (tabCurrent && tabSaved) {
  tabCurrent.addEventListener('click', () => {
    // Force show Current panel, hide Saved panel
    if (toolPanel) toolPanel.style.display = '';     // Restore display
    if (savedPanel) savedPanel.style.display = 'none'; // Force hide
    
    tabCurrent.classList.add('active');
    tabSaved.classList.remove('active');
  });

  tabSaved.addEventListener('click', () => {
    // Force hide Current panel, show Saved panel
    if (toolPanel) toolPanel.style.display = 'none';   // Force hide
    if (savedPanel) savedPanel.style.display = '';     // Restore display
    
    tabSaved.classList.add('active');
    tabCurrent.classList.remove('active');
  });
}

document.getElementById('tabCurrent').addEventListener('click', () => {
  document.getElementById('tool').classList.remove('hidden');
  document.getElementById('savedPanel').classList.add('hidden');
  document.getElementById('tabCurrent').classList.add('active');
  document.getElementById('tabSaved').classList.remove('active');
});

document.getElementById('tabSaved').addEventListener('click', () => {
  document.getElementById('tool').classList.add('hidden');
  document.getElementById('savedPanel').classList.remove('hidden');
  document.getElementById('tabSaved').classList.add('active');
  document.getElementById('tabCurrent').classList.remove('active');
});

// 2. Export and Save functionality
document.getElementById('exportResultBtn').addEventListener('click', () => {
  if (!current_huff_result) return alert('No current result to export. Please run a prediction first.');
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(current_huff_result, null, 4));
  const downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", `huff_result_${new Date().getTime()}.json`);
  document.body.appendChild(downloadAnchorNode);
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
});

document.getElementById('saveResultBtn').addEventListener('click', () => {
  if (!current_huff_result) return alert('No result to save.');
  
  // Clone the data and append a timestamp
  const savedItem = JSON.parse(JSON.stringify(current_huff_result));
  savedItem.save_time = new Date().toLocaleString();
  savedItem.id = new Date().getTime(); // Unique ID
  
  result_array.push(savedItem);
  renderSavedList();
  alert('Result saved successfully!');
});

// 3. Render the saved list
function renderSavedList() {
  const container = document.getElementById('savedListContainer');
  const emptyMsg = document.getElementById('noSavedMsg');
  
  if (result_array.length === 0) {
    emptyMsg.style.display = 'block';
    container.classList.add('hidden');
    return;
  }
  
  emptyMsg.style.display = 'none';
  container.classList.remove('hidden');
  container.innerHTML = '';
  
  result_array.forEach(item => {
    const card = document.createElement('div');
    card.className = 'saved-item-card';
    card.innerHTML = `
      <div class="saved-item-info">
        <p><strong>Category:</strong> ${item.matched_category || 'N/A'} (NAICS: ${item.naics_code || 'N/A'})</p>
        <p><strong>Location:</strong> Lat: ${item.candidate_lat}, Lon: ${item.candidate_lon}</p>
        <p><strong>Floor Area:</strong> ${item.floor_area} sq m</p>
        <p style="font-size: 0.8rem; color: #888;"><strong>Saved at:</strong> ${item.save_time}</p>
      </div>
      <div class="saved-item-actions">
        <button class="btn-compare" onclick="openCompareModal(${item.id})">Compare</button>
        <button class="btn-delete" onclick="deleteSavedItem(${item.id})" title="Delete">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" style="vertical-align: middle;">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    `;
    container.appendChild(card);
  });
}

// 4. Delete logic
window.deleteSavedItem = function(id) {
  if (confirm("Are you sure you want to delete this result?")) {
    result_array = result_array.filter(item => item.id !== id);
    renderSavedList();
  }
};

// 5. Import functionality (Drag & Drop and File Selection)
const importModal = document.getElementById('importModal');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

document.getElementById('importDataBtn').addEventListener('click', () => {
  importModal.classList.remove('hidden');
});
document.getElementById('closeImportBtn').addEventListener('click', () => {
  importModal.classList.add('hidden');
});
document.getElementById('browseFileText').addEventListener('click', () => {
  fileInput.click();
});

function handleFile(file) {
  if (!file || file.type !== "application/json") {
    return alert("Please upload a valid JSON file.");
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const parsed = JSON.parse(e.target.result);
      
      // Support direct object or nested within the 'huff_result' field
      const data = parsed.huff_result ? parsed.huff_result : parsed;
      
      // Basic format validation
      if (data && data.candidate_lat && data.floor_area) {
        data.save_time = new Date().toLocaleString();
        data.id = new Date().getTime();
        result_array.push(data);
        renderSavedList();
        alert("File imported successfully!");
        importModal.classList.add('hidden');
      } else {
        alert("Invalid format: Missing required fields (e.g. candidate_lat).");
      }
    } catch (err) {
      alert("Error parsing JSON file.");
    }
  };
  reader.readAsText(file);
}

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

// 6. Chart comparison logic
const compareModal = document.getElementById('compareModal');
document.getElementById('closeCompareBtn').addEventListener('click', () => {
  compareModal.classList.add('hidden');
});

window.openCompareModal = function(savedId) {
  if (!current_huff_result) {
    return alert("You don't have a 'Current' prediction result to compare against. Please run the model first.");
  }
  
  const savedItem = result_array.find(item => item.id === savedId);
  if (!savedItem) return;

  compareModal.classList.remove('hidden');
  const ctx = document.getElementById('compareChartCanvas').getContext('2d');
  
  if (compareChartInst) {
    compareChartInst.destroy();
  }

  // Render the comparison chart. Due to the large variance in metrics (e.g., area 4000 vs market share 0.02), the Y-axis uses a logarithmic scale.
  compareChartInst = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Market Share', 'Total Demand', 'Predicted Visits', 'Floor Area'],
      datasets: [
        {
          label: 'Current Result',
          data: [
            current_huff_result.market_share,
            current_huff_result.total_demand,
            current_huff_result.predicted_visits,
            current_huff_result.floor_area
          ],
          backgroundColor: 'rgba(37, 99, 235, 0.8)', // Action Blue
          borderRadius: 4
        },
        {
          label: 'Saved Result',
          data: [
            savedItem.market_share,
            savedItem.total_demand,
            savedItem.predicted_visits,
            savedItem.floor_area
          ],
          backgroundColor: 'rgba(14, 165, 164, 0.8)', // Pull Teal
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': ' + context.raw;
            }
          }
        }
      },
      scales: {
        y: {
          type: 'logarithmic',
          title: {
            display: true,
            text: 'Value (Log Scale)'
          }
        }
      }
    }
  });
};