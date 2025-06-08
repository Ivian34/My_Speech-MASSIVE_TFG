# My_Speech_Massive_TFG

> **Desenvolupament d’un model de comprensió de la parla per a videojocs**  
> Treball de Fi de Grau per al Grau en Enginyeria Informàtica (Computació)  
> Autor: Iván Cobos Navarro  
> Director: Francisco Javier Hernando Pericas  
> Codirector: Alexandre Cristian Peiró Lilja  

---

## 📋 Descripció

Aquest repositori conté el codi i la documentació del TFG en el qual s’adapta i amplia el model Whisper d’OpenAI per a la tasca de **Spoken Language Understanding (SLU)** en Català. El model transcriu àudio i, a més, reconeix intencions i omple “slots” tant del tipus BIO com altres slots específics (per exemple: player, piece, location_x, location_y).

---

## Entorns principals

Les següents configuracions han estat provades i suporten les versions indicades, però no es garanteix compatibilitat amb d’altres:

- Python 3.9.4  
- HuggingFace Transformers 4.37  

### Entrenament del model E2E SLU amb el dataset propi de prova del TFG

> **Avís:** els exemples següents funcionen amb la llibreria HuggingFace.

1. **Clonar el repositori a la ruta desitjada**  
    ```bash
    git clone https://github.com/Ivian34/My_Speech-MASSIVE_TFG /la/teva/ruta/speech-massive-code
    ```

2. **Crear i activar el teu entorn virtual**  
    - Amb `venv`:
      ```bash
      python -m venv /ruta/al/teu/entorn
      source /ruta/al/teu/entorn/bin/activate
      ```
    - Amb `conda`:
      ```bash
      conda create --name el-meu-entorn python=3.9
      conda activate el-meu-entorn
      ```

3. **Actualitzar pip per suportar `pyproject.toml`**  
    ```bash
    pip install --upgrade pip
    ```

4. **Instal·lar el projecte Speech-MASSIVE en mode editable**  
    ```bash
    pip install -e .
    ```

5. **Revisar i modificar els fitxers d’hiperparàmetres**  
    ```text
    .
    └── src/speech_massive/examples/hparams
        ├── e2e_slu_zeroshot_fr.yaml    

6. **Entrenament** 

python3 ./src/speech_massive/examples/speech/run_slu_whisper_test.py ./src/speech_massive/examples/hparams/e2e_slu_zeroshot_fr.yaml

## 🤝 Autors i crèdits

Aquest projecte es basa en el treball original de **Beomseok Lee, Ioan Calapodescu, Marco Gaido, Matteo Negri i Laurent Besacier** i col·laboradors de [Speech-MASSIVE](https://github.com/hlt-mt/Speech-MASSIVE) :contentReference[oaicite:0]{index=0}  
Les dades d’entrenament i validació són de creació pròpia i no estan subjectes a límits de distribució externs.
