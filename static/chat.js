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
  const section = document.getElementById("chartSection");
  if (!section) return;

  // Hide and tear down the chart if there is no data.
  if (!competitors || competitors.length === 0) {
    section.classList.add("hidden");
    if (competitorsChart) {
      competitorsChart.destroy();
      competitorsChart = null;
    }
    return;
  }

  section.classList.remove("hidden");

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
      const label = `Your store (predicted): ${candidateRefVisits.toFixed(1)}/day`;
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
          beginAtZero: true,
          title: { display: true, text: "Visits/day" },
          grid: { color: "rgba(15, 23, 42, 0.06)" }
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
