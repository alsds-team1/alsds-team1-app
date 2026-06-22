const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

// Declared at top level so we can destroy the old chart instance before redrawing.
let competitorsChart = null;

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
      updateCompetitorsChart(r.competitors);

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
  const visitsText = (visits === null || visits === undefined || visits === "") ? "N/A" : String(visits);
  const marketShare = Number(result.market_share);
  const sharePct = Number.isFinite(marketShare) ? (marketShare * 100).toFixed(2) : null;
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
    </div>
    <div class="result-meta">
      ${floorArea ? `<span><strong>Floor area:</strong> ${escapeHtml(floorArea)} m²</span>` : ""}
      <span><strong>Runtime:</strong> ${escapeHtml(runtime)} ms</span>
    </div>
    ${notes ? `<div class="result-note">${escapeHtml(notes)}</div>` : ""}
  `;

  const competitors = Array.isArray(result.competitors) ? result.competitors : [];

  if (competitors.length === 0) {
    tableWrap.innerHTML = "No competitor records returned.";
    return;
  }

  tableWrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Distance (mi)</th>
          <th>Visits/day</th>
          <th>Attraction (capture prob.)</th>
        </tr>
      </thead>
      <tbody>
        ${competitors.map(c => `
          <tr>
            <td>${escapeHtml(c.name ?? c.place_name ?? c.poi_name ?? "Unknown")}</td>
            <td>${escapeHtml(c.distance_miles ?? c.distance ?? "N/A")}</td>
            <td>${escapeHtml(c.size ?? c.floor_area ?? c.area ?? "N/A")}</td>
            <td>${escapeHtml(c.attraction ?? "N/A")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function updateCompetitorsChart(competitors) {
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

  // Sort by attraction descending and take the top 10.
  const top10 = [...competitors]
    .sort((a, b) => (Number(b.attraction) || 0) - (Number(a.attraction) || 0))
    .slice(0, 10);

  const labels = top10.map(c => c.name ?? c.poi_name ?? "Unknown");
  const data = top10.map(c => Number(c.attraction) || 0);

  // If the chart already exists, update its data in place so the bars animate
  // smoothly to the new values (instead of being destroyed and rebuilt).
  if (competitorsChart) {
    competitorsChart.data.labels = labels;
    competitorsChart.data.datasets[0].data = data;
    competitorsChart.update();
    return;
  }

  const ctx = document.getElementById("topCompetitorsChart").getContext("2d");

  competitorsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Attraction Score",
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
            label: (item) => `Attraction: ${Number(item.parsed.y).toFixed(4)}`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "Attraction Score" },
          grid: { color: "rgba(15, 23, 42, 0.06)" }
        },
        x: {
          grid: { display: false },
          ticks: { maxRotation: 45, minRotation: 30, autoSkip: false, font: { size: 10 } }
        }
      }
    }
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
