# ⚡ PhotoSender AI Solution (SmartKeywords & Vision Engine)

Motore intelligente standalone per fotogiornalismo sportivo, ottimizzato per **Mac Apple Silicon (M3 / Metal Performance Shaders)** e architetture Client-Server con supporto IMAGO, agenzie internazionali e workflow Photo Mechanic.

---

## 🎯 Obiettivi Chiave
1. **Riconoscimento Automatico della Tipologia Fotografica**:
   - `Match Action` (contrasti, dribbling, tiri, passaggi)
   - `Goal Celebration` (esultanze singole e di gruppo)
   - `Goalkeeper Action` (parate, tuffi, rinvii)
   - `Bench & Coach` (allenatore, staff, reazioni)
   - `Player Portrait / Reaction` (primi piani, emozioni, sguardi)
   - `Supporters & Atmosphere` (tifosi, curve, coreografie, fumogeni)
   - `Warm-up / Pre-match` (riscaldamento)
2. **Rilevamento Volti & Inquadratura**:
   - Conteggio volti, framing (Close-up, Medium, Wide), verifica nitidezza (Quality Check).
3. **Associazione Editoriale Giocatore al Database**:
   - Rilevamento numero di maglia (YOLO body crop + OCR).
   - Incrocio in tempo reale con i giocatori e le rose della partita (`roster_player`, `squadre`).
   - Nessun confronto biometrico facciale: associazione basata su numero, squadra, partita e conferma editoriale.
4. **Generazione Automatica Didascalie (Caption) in Inglese**:
   - Sintassi rigorosa conforme allo standard editoriale internazionale / IMAGO:
     `[Player] of [Team] in action / celebrates during the [Competition] match between [Home] and [Away] at [Venue] in [City], [Country]...`
5. **Generazione SmartKeywords IMAGO**:
   - Interrogazione del database unificato (`keyword_sets`, `keyword_triggers`, `keyword_set_terms`, `agency_keyword_profiles`).
6. **Scrittura Metadati IPTC / XMP**:
   - Wrapper ExifTool con flag `-P` per preservare al 100% data e ora di scatto originale e convenzione naming `SP24_...`.
7. **Control Panel di Revisione 1-Click (Desk Partita)**:
   - Anteprima visiva e bottoni rapidi per selezionare giocatori o cambiare azioni all'istante, pronto per l'invio FTP.

---

## 🏗️ Architettura & Integrazione con PhotoSender
- **Database condiviso**: MySQL locale / VPS (`photoftpnewtesting`).
- **Nessuna dipendenza pesante per i portatili allo stadio**:
  - Gira come microservizio REST API (FastAPI) accessibile da web browser o integrato nel frontend React di PhotoSender.
  - Sull'iMac M3 sfrutta l'accelerazione GPU Apple MPS (`torch.device("mps")`).

