# E2E-R
Code for the article [Fine-tuning Self-Supervised Learning Models for End-to-End Pronunciation Scoring](https://ieeexplore.ieee.org/document/10255657).

This library includes code for training an end-to-end pronunciation scoring model. 

This code was built using [SpeechBrain](https://speechbrain.github.io/).

To run the experiments, follow the following steps:
1. Install the requirements by running `pip install -r requirements.txt`
2. (Optional) [Install Kaldi](https://kaldi-asr.org/doc/install.html). This is only necessary if you would like to run the LSTM scorer experiment.
3. Place the TIMIT and speechocean762 datasets in the desired data directory.
4. Set the variable `DATA_DIR` in the `run.sh` file to the path of the data directory.
5. Run the experiments in `run.sh` file.

N.B.: We suggest running the experiments in the `run.sh` file by copying the commands from the file and pasting them into the terminal for easier debugging.

Cite as:
```
@article{zahran2023fine,
  title={Fine-tuning Self-Supervised Learning Models for End-to-End Pronunciation Scoring},
  author={Zahran, Ahmed and Fahmy, Aly and Wassif, Khaled and Bayomi, Hanaa},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```