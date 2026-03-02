# OGPR Browse App

Applicazione **Streamlit** per la visualizzazione e l'elaborazione di file `.ogpr` (Ground Penetrating Radar).

## Funzionalita

- Import multiplo di file `.ogpr` con auto-detect layout e offset
- Pipeline di filtri configurabile (Gain, Bandpass, Background removal, Hilbert, Dewow, Smoothing)
- Modalita workflow: **Base consigliato** e **Manuale** (ordini custom)
- Suggerimento automatico preset via **Machine Learning** (scikit-learn)
- Visualizzazione radargrammi con scale colore personalizzabili
- Generazione **time-slice** con supporto coordinate GPS

## Struttura del progetto

```
ogpr-browse-app/
├── app.py                  # Entry point Streamlit
├── requirements.txt
├── README.md
├── .gitignore
│
├── gpr_app/                # Pacchetto core applicazione
│   ├── models.py           # Dataclass: FilterConfig, VisualConfig, DecodeConfig, CoordinateConfig
│   ├── constants.py        # Costanti: COLOR_SCALES, LAYOUT_OPTIONS, FILTER_STATE_DEFAULTS
│   ├── pipeline.py         # Pipeline filtri GPR
│   ├── utils.py            # Utilita generali (to_float, sample_rate_from_metadata)
│   └── ui/
│       └── sidebar.py      # Costruttori sidebar Streamlit
│
├── radar_filters.py        # Filtri segnale GPR (scipy)
├── radar_io.py             # I/O file OGPR
├── ml_presets.py           # Preset ML per selezione automatica filtri
└── ui_views.py             # Viste principali (profili, time-slice)
```

## Installazione

```bash
pip install -r requirements.txt
```

## Avvio

```bash
streamlit run app.py
```

## Dipendenze principali

| Libreria | Uso |
|---|---|
| `streamlit` | Interfaccia web |
| `numpy` | Elaborazione numerica |
| `scipy` | Filtri segnale (butter, hilbert, gaussian) |
| `plotly` | Grafici interattivi |
| `h5py` | Lettura file OGPR (HDF5) |
| `scikit-learn` | Suggerimento automatico preset |
