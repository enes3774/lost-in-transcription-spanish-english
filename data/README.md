# data/

Not in git. Inference reads a data directory in the competition's layout; training reads the raw Bangor Miami download.

```
data/
├── raw/
│   ├── miami/                 <- Bangor Miami corpus as downloaded from the Mozilla Data Collective
│   │   ├── chat/*.cha             (56 CHAT transcripts)
│   │   ├── audios/*.mp3           (56 recordings)
│   │   └── word_level_tsvs/, Miami_doc.pdf
│   └── dev/                   <- the competition dev set: clips/, test_metadata.csv, refs.csv (measurement only)
└── test/                      <- any set to transcribe, in the competition layout:
    ├── clips/<audio file>
    └── test_metadata.csv          (columns audio_filename, file_duration_seconds, language)
```

For the official test or dev set, unzip the organizers' download so it looks like `data/test/`. For your own audio,
`python src/inference/prepare_data.py <folder_of_audio> data/test` builds the layout. `src/training/README.md` §1 turns
`data/raw/miami` into the 22,290 training clips and the speaker-disjoint split.
