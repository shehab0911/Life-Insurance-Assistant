(() => {
  const micBtn = document.getElementById("mic");
  const statusEl = document.getElementById("status");
  const transcriptEl = document.getElementById("transcript");

  const card = document.querySelector(".card");
  const resetBtn = document.createElement("button");
  resetBtn.textContent = "Reset Session";
  resetBtn.style.marginTop = "8px";
  resetBtn.onclick = () => {
    ws.send(JSON.stringify({ type: "reset", session_id: sessionId }));
    transcriptHistory = [];
    renderHistory();
  };
  card.appendChild(resetBtn);

  const histDiv = document.createElement("div");
  histDiv.style.marginTop = "10px";
  histDiv.style.fontSize = "12px";
  card.appendChild(histDiv);

  const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = wsProtocol + "://" + location.host + "/ws";
  const ws = new WebSocket(wsUrl);

  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  const sessionId = "session_" + Math.random().toString(36).slice(2, 9);
  let transcriptHistory = [];

  ws.onopen = () => {
    console.log("WebSocket open");
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "transcript") {
      transcriptEl.textContent = "You said: " + msg.text;
      statusEl.textContent = "Processing...";
      transcriptHistory.push({ who: "user", text: msg.text });
    } else if (msg.type === "response") {
      statusEl.textContent = "Assistant replied (speaking)...";
      transcriptEl.textContent = "";
      transcriptHistory.push({ who: "assistant", text: msg.text });
      speakText(msg.text);
    } else if (msg.type === "error") {
      statusEl.textContent = "Error: " + msg.message;
    }
    renderHistory();
  };

  function renderHistory() {
    histDiv.innerHTML = transcriptHistory
      .map(
        (it) =>
          `<div style="margin-bottom:6px"><strong>${it.who}:</strong> ${it.text}</div>`
      )
      .join("");
  }

  function speakText(text) {
    if (!("speechSynthesis" in window)) {
      statusEl.textContent =
        "No TTS supported in your browser. Showing text: " + text;
      return;
    }
    const utter = new SpeechSynthesisUtterance(text);
    utter.onend = () => {
      statusEl.textContent = "Ready — ask another question.";
    };
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };
      mediaRecorder.onstop = onRecordingStop;
      mediaRecorder.start();
      isRecording = true;
      micBtn.textContent = "⏺";
      statusEl.textContent = "Recording... Speak now.";
    } catch (err) {
      statusEl.textContent = "Microphone access denied.";
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      micBtn.textContent = "🎤";
      statusEl.textContent = "Uploading audio...";
    }
  }

  async function onRecordingStop() {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    const arrayBuffer = await blob.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(
        null,
        bytes.subarray(i, i + chunkSize)
      );
    }
    const b64 = btoa(binary);
    const dataUri = "data:audio/webm;base64," + b64;

    ws.send(
      JSON.stringify({ type: "audio", data: dataUri, session_id: sessionId })
    );
    statusEl.textContent = "Sent audio, waiting for transcript...";
  }

  micBtn.addEventListener("click", () => {
    if (!isRecording) startRecording();
    else stopRecording();
  });
})();
