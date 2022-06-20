Cross-Lingual-Voice-Cloning


**DISCLAIMER :- Based on the paper [Learning to Speak Fluently in a Foreign Language:Multilingual Speech Synthesis and Cross-Language Voice Cloning](https://arxiv.org/pdf/1907.04448.pdf)**
<br />Differences from original fork:
* cleaned tensorflow requirement;
* added russian language support;
* various code improvements
## Dataset Format
The model needs to be provided 2 text files 1 for the purpose of training and 1 for validation. Each line of the txt file should follow the following format :- 
```
<path-to-wav-file>|<text-corresponding-to-speech-in-wav>|<speaker-no>|<lang-no>
```

```<speaker-no>``` goes from 0 to n-1, where n is the number of speakers.

```<lang-no>``` goes from 0 to m-1 , where m is the number of languages. <br>
Lnaguage-id table:
* en : 0
* ru : 1

## Hparams
```hparams.training_files, hparams.validation_files``` need to be set to the path to the txt files of previous section.

```hparams.n_speakers, hparams.dim_yo``` need to be changed to the number of speakers.

```hparams.n_langs``` must be set to number of languages.

To change the languages, add/remove unicode characters in ```_letters``` variable of ```text/symbols.py``` .

## Inference

For inference using the model , run clvc-infer-gh.ipynb with appropriate speaker and language number.
