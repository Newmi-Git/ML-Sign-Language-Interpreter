# SASL Recognition (prototype)

Real-time recognition of South African Sign Language fingerspelling and a small
vocabulary of word signs, running from a webcam in the browser.

![demo](demo.gif)

## What it does
- Spells letters from hand shape (MLP) and J/Z from motion (GRU)
- Recognises N word signs: HELLO, THANK-YOU, ... (list)
- Strings signs into glosses and optionally turns them into an English sentence
- Video stays in the browser; only hand/body landmarks go to the server

## How it works
MediaPipe hand + pose landmarks -> normalised features (hand shape, body-relative
position) -> 30-frame clips -> GRU classifier (PyTorch) -> Flask API -> Vue 3 app.

## Results
Tested on sessions the model never saw during training.
| Model | Classes | Test accuracy |
|---|---|---|
| Letters | 24 | XX% |
| Words | N | XX% |

## Limitations
Trained on data from N signer(s). Accuracy drops for new signers and lighting.
This is a research prototype, not an interpreter. SASL has its own grammar and
facial markers that this does not capture. Vocabulary reviewed by: (name / group, only if true).

## Run it
```
pip install -r requirements.txt
cd data
python collect_words.py   # record signs
python train_words.py     # train
python server.py          # API on :5000
cd ../web && npm install && npm run dev
```