# OGPR Browse App

Web app (Streamlit) per dati GPR in formato `.ogpr`:
- import multiplo di file OGPR (anche un file per canale antenna, es. stream DP IDS)
- visualizzazione dei profili radar per ogni file/canale
- creazione time-slice interpolata usando le coordinate presenti nei file OGPR
- applicazione filtri (gain, band-pass, background removal, DC removal, attenuation correction, hilbert, cutTWT, dewow, smoothing)
- processing multicanale e geometrico (channel shift, exchange X/Y, trace distance costante)
- interpolazione time-slice anche con IDW e mappa coherence

## Avvio rapido

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Flusso nell'app

1. Carica tutti i file `.ogpr` della survey (anche in batch).
2. In `Sezione = Profili` seleziona il profilo da ispezionare.
3. In `Sezione = Time-slice` scegli il sample e genera la mappa interpolata sulle coordinate.
4. Se il radargramma appare piatto, usa la sidebar `Decoding OGPR` per provare `Offset mode` e `Layout volume radar`.
5. Usa `Filtri` con ordine esplicito (`0 = off`) per replicare la logica a step del plugin MATLAB.
6. In `Filtri`, scegli `Workflow filtri`: `Base consigliato` (ordine automatico) oppure `Manuale` (ordini personalizzati). Ogni filtro ha switch `Abilita ...` ON/OFF.
7. In `Filtri`, usa `Preset tecnico OGPR (metadata-driven)` per impostare band-pass e parametri tecnici in funzione dei metadati del file (sample rate, frequenza antenna, ecc.; con fallback su stima spettrale).
8. Usa `Coordinate` per filtro punti fermi e smoothing mediano delle coordinate prima del binning.
9. In `Profili`, usa `Diagnostica filtri (step-by-step)` per verificare dove il segnale si riduce.
10. In `Profili`, scegli la modalita di visualizzazione (`Solo filtrato`, `Grezzo + filtrato`, `Solo grezzo`): il grezzo e mostrato solo se selezionato.
11. In `Preset ML automatico`, premi prima `Calcola suggerimento ML` (on-demand) e poi `Applica preset ML`.
12. Usa il selettore `Sezione` (`Profili` / `Time-slice`) per elaborare una sola vista per volta e ridurre i tempi di attesa.

## Filtri disponibili

- `Gain (gain in dB lungo asse temporale)`
- `Band-pass` (`butter` e `gpr_corner`)
- `Rimozione background`
- `DC removal`
- `Attenuation correction`
- `Hilbert` (`envelope`, `real`, `imag`, `phase`)
- `cutTWT`
- `Dewow`
- `Smooth gaussiano`

## Processing aggiuntivi

- `channelshift` (allineamento t0 tra canali, riferimento CH1)
- `exchange_x_y` (scambio coordinate X/Y)
- `constTraceDist` (resampling profilo a passo costante)
- `coherence` (mappa coherence su finestra temporale)
- `idw3dblock` (interpolazione IDW per time-slice)

## Note formato `.ogpr`

In questa versione `.ogpr` supporta:
- OpenGPR nativo (`ogpr` magic + header JSON + data block radar)
- `.ogpr` HDF5-based
- value type OpenGPR: `int16`, `uint16`, `float32`, `float64`
- lettura del blocco `Sample Geolocations` per usare coordinate X/Y (GPS o CRS definito nel file)
- auto-detect robusto di `byteOffset` (assoluto/relativo) e layout del volume radar

Se il tuo `.ogpr` usa una variante diversa, bisogna aggiungere il parser specifico in `radar_io.py`.
