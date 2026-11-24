<script setup>
import { ref } from "vue"
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision"

const API = import.meta.env.VITE_API || "http://localhost:5000"
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
const WINDOW_MS = 2000   // must match REC_SECONDS in collect_words.py (1.0 s)

// what to show/say for each sign. Edit to match your vocabulary.
const SAY = {
  "HELLO": "Hello",
  "HOW-ARE-YOU": "How are you?",
  "FINE": "I'm fine",
  "THANK-YOU": "Thank you",
  "PLEASE": "Please",
  "SORRY": "Sorry",
  "HELP": "Help me, please",
  "WATER": "I need water",
  "TOILET": "Where is the toilet?",
  "YES": "Yes",
  "NO": "No",
}

const video = ref(null)
const canvas = ref(null)
const status = ref("Press Start")
const glosses = ref([])

let hands, pose
let buffer = []          // the last second of frames: { t, frame }
let lastSend = 0
let lastLabel = null
let lastAccepted = 0

async function start() {
  try {
    status.value = "Loading models..."
    const fs = await FilesetResolver.forVisionTasks(WASM)
    hands = await HandLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: "/models/hand_landmarker.task" },
      runningMode: "VIDEO",
      numHands: 2,
    })
    pose = await PoseLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: "/models/pose_landmarker.task" },
      runningMode: "VIDEO",
    })
    video.value.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
    })
    await video.value.play()
    canvas.value.width = video.value.videoWidth
    canvas.value.height = video.value.videoHeight
    status.value = "Sign something"
    requestAnimationFrame(loop)
  } catch (e) {
    status.value = "Error: " + e.message
  }
}

const pts = (l) => l.map((p) => [p.x, p.y, p.z])

function dots(ctx, lms, W, H, colour) {
  ctx.fillStyle = colour
  for (const p of lms) {
    ctx.beginPath()
    ctx.arc(p.x * W, p.y * H, 3, 0, 7)
    ctx.fill()
  }
}

function loop() {
  const c = canvas.value
  const W = c.width
  const H = c.height
  const ctx = c.getContext("2d")

  // draw the webcam mirrored (same as cv2.flip in the Python scripts)
  ctx.save()
  ctx.scale(-1, 1)
  ctx.drawImage(video.value, -W, 0, W, H)
  ctx.restore()

  const t = performance.now()
  const h = hands.detectForVideo(c, t)
  const p = pose.detectForVideo(c, t)

  const frame = { left: null, right: null, pose: p.landmarks[0] ? pts(p.landmarks[0]) : null }
  h.landmarks.forEach((lm, i) => {
    frame[h.handedness[i][0].categoryName.toLowerCase()] = pts(lm)
    dots(ctx, lm, W, H, "#22d3ee")
  })

  // keep only the last second, and ask the server about it 4 times a second
  buffer.push({ t, frame })
  buffer = buffer.filter((b) => t - b.t <= WINDOW_MS)
  if (h.landmarks.length && t - lastSend > 250 && buffer.length >= 10) {
    lastSend = t
    send(buffer.map((b) => b.frame))
  }
  requestAnimationFrame(loop)
}

async function send(frames) {
  try {
    const r = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    })
    const { label, conf } = await r.json()
    status.value = label ? `${label} (${conf.toFixed(2)})` : "..."
    const now = performance.now()
    if (label && label !== "NONE" && conf >= 0.8) {
      // accept a sign when it is the answer twice in a row, then wait 1.5 s
      if (label === lastLabel && now - lastAccepted > WINDOW_MS + 500) {
        glosses.value.push(label)
        lastAccepted = now
        lastLabel = null
      } else {
        lastLabel = label
      }
    } else {
      lastLabel = null
    }
  } catch (e) {
    status.value = "Server not reachable"
  }
}

function speak() {
  const text = glosses.value.map((g) => SAY[g] || g).join(". ")
  speechSynthesis.speak(new SpeechSynthesisUtterance(text))
}

function clearAll() {
  glosses.value = []
}
</script>

<template>
  <main class="wrap">
    <h1>SASL Recognition (prototype)</h1>
    <video ref="video" hidden playsinline muted></video>
    <canvas ref="canvas" width="640" height="480"></canvas>
    <p class="status">{{ status }}</p>
    <button @click="start">Start</button>

    <h2>Signs</h2>
    <p class="glosses">{{ glosses.map((g) => SAY[g] || g).join(" · ") || "-" }}</p>
    <button @click="speak" :disabled="!glosses.length">Speak</button>
    <button @click="clearAll">Clear</button>

    <p class="note">Your video never leaves your device. Only hand and body points are sent to the server.</p>
  </main>
</template>

<style scoped>
.wrap { max-width: 680px; margin: 0 auto; padding: 1rem; font-family: system-ui, sans-serif; }
canvas { width: 100%; border-radius: 8px; background: #111; }
button { margin: 0 0.5rem 0.5rem 0; padding: 0.5rem 1rem; }
.glosses { font-size: 1.4rem; font-weight: 600; }
.note { font-size: 0.8rem; opacity: 0.7; }
</style>