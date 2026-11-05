# ML Sign Language Interpreter

Live interpretation of fingerspelling via a webcam feed. It detects your hand using MediaPipe, identifies letters via compact PyTorch networks, and progressively assembles text on the display.

![demo](demo.gif)

## How it works

Two models work together:

1. **Still letters (A-Y, excluding J):** hand landmarks are adjusted (shifted to align at the wrist, rescaled relative to hand dimensions) and passed to a modest multi-layer perceptron.
2. **Dynamic letters (J and Z):** because these involve motion, a GRU analyzes a 30-frame sequence of hand landmarks together with wrist displacement. A `NONE` category prevents triggering when the hand is merely stationary.

`live.py` applies temporal smoothing to predictions across recent frames, imposes a confidence cutoff, and only inserts a character after you maintain the gesture momentarily. Dropping your hand inserts a blank space.

## Results

- Static letter model validation accuracy: **XX%**
- Motion letter model validation accuracy: **XX%**

Models were developed using roughly 28,600 static examples and approximately 600 motion sequences from one individual. Since the validation set is randomly selected, these figures probably exceed performance with a different person or camera setup.

## Project structure

```
data/
  collect.py       capture static letter examples
  collect_seq.py   capture motion sequences (J, Z, NONE)
  features.py      landmark adjustment
  model.py         MLP and GRU network definitions
  train.py         develop the static model
  train_seq.py     develop the motion model
  live.py          execute the real-time interpreter
  models/          MediaPipe downloads and network parameters
```

## Setup

1. Set up Python 3.12 and connect a webcam.
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Operate from within the `data` directory:
   ```
   cd data
   python live.py
   ```

Pre-trained parameters are provided, so operation does not require any model development.

## Controls (live.py)

- `q`: exit
- `space`: insert a space
- `backspace`: remove the most recent character
- `c`: erase the entire sentence

## Training your own data

1. Gather static letters. Display a character and press its corresponding key to store an example (`1` exits):
   ```
   python collect.py
   ```
2. Gather motion sequences. Press `j`, `z`, or `0` (no movement) and then perform the gesture (`1` exits):
   ```
   python collect_seq.py
   ```
3. Develop models:
   ```
   python train.py
   python train_seq.py
   ```

## Limitations

- The model learns from a single individual, so its precision might diminish for different hands, skin colors, lighting setups, and camera perspectives.
- It recognizes finger spelling exclusively. The system does not interpret complete signs or full sentences.
- Class sample numbers vary considerably, ranging from roughly 640 examples per letter up to around 1,970.

## Built with

Python, MediaPipe, OpenCV, PyTorch, scikit-learn