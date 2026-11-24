<script setup>
import { ref } from "vue"
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision"

const API = import.meta.env.VITE_API || "http://localhost:5000"
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"

const video = ref(null)
const canvas = ref(null)
const status = ref("Press Start")
const glosses = ref([])
const sentence = ref("")

let hands, pose
let clip = []      // frames of the sign being made right now
let missed = 0     // frames in a row with no hand

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
    status.value = "Sign a word, then lower your hands"
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

  if (h.landmarks.length) {
    clip.push(frame)
    missed = 0
  } else if (clip.length && ++missed > 8) {   // hands gone for ~8 frames = sign finished
    send(clip)
    clip = []
    missed = 0
  }
  if (clip.length > 90) {                      // safety: cut very long clips
    send(clip)
    clip = []
  }
  requestAnimationFrame(loop)
}

async function send(frames) {
  if (frames.length < 10) return
  try {
    const r = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    })
    const { label, conf } = await r.json()
    status.value = label ? `Heard ${label} (${conf.toFixed(2)})` : "Too short"
    if (label && label !== "NONE" && conf >= 0.8) glosses.value.push(label)
  } catch (e) {
    status.value = "Server not reachable"
  }
}

async function makeSentence() {
  const r = await fetch(`${API}/sentence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ glosses: glosses.value }),
  })
  sentence.value = (await r.json()).text
}

function clearAll() {
  glosses.value = []
  sentence.value = ""
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
    <p class="glosses">{{ glosses.join("  ") || "-" }}</p>
    <button @click="makeSentence" :disabled="!glosses.length">Make sentence</button>
    <button @click="clearAll">Clear</button>
    <p class="sentence">{{ sentence }}</p>

    <p class="note">Your video never leaves your device. Only hand and body points are sent to the server.</p>
  </main>
</template>

<style scoped>
.wrap { max-width: 680px; margin: 0 auto; padding: 1rem; font-family: system-ui, sans-serif; }
canvas { width: 100%; border-radius: 8px; background: #111; }
button { margin: 0 0.5rem 0.5rem 0; padding: 0.5rem 1rem; }
.glosses { font-size: 1.4rem; font-weight: 600; }
.sentence { font-size: 1.2rem; }
.note { font-size: 0.8rem; opacity: 0.7; }
</style>