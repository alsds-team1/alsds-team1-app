const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

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

  try {
    /*
      IMPORTANT:
      Before treating the message as a normal follow-up question,
      check whether the user is asking to rerun the model with a new full set of inputs.

      Example supported message:
      "use 42.229212, -71.805525 and rerun the model for NAICS code 4441 and area of 1000 square meters"
    */
    const rerunInputs = extractRerunInputs(text);

    if (rerunInputs) {
      await rerunModelFromMessage(rerunInputs);
      return;
    }

    if (state.step === "category") {
      const response = await fetch("/api/trans_naics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: text.trim() })
      });

      const data = await response.json();

      if (!data.ok) {
        addBotMessage(data.error || "I'm sorry, I couldn't identify that business type. Please try again with a more specific industry description, such as 'Beer, Wine, and Liquor Stores', 'Bakeries and Tortilla Manufacturing', or 'Building Material and Supplies Dealers'.");
        return;
      }

      state.business_category = data.naics_code;
      state.step = "location";

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

  const predictedVisits = result.predicted_visits ?? "N/A";
  const marketShare = Number(result.market_share);
  const runtime = result.runtime_ms ?? "N/A";
  const notes = result.notes ?? "";

  summary.innerHTML = `
    <strong>Predicted Visits:</strong> ${escapeHtml(predictedVisits)}<br>
    <strong>Estimated Market Share:</strong> ${Number.isFinite(marketShare) ? (marketShare * 100).toFixed(2) + "%" : "N/A"}<br>
    <strong>Runtime:</strong> ${escapeHtml(runtime)} ms<br>
    <strong>Notes:</strong> ${escapeHtml(notes)}
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
          <th>Distance</th>
          <th>Size</th>
          <th>Attraction</th>
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
