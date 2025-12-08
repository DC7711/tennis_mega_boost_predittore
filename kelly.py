import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# 1. CONFIGURAZIONE PAGINA
# ==============================================================================
st.set_page_config(
    page_title="Tennis AI Assistant",
    page_icon="🎾",
    layout="centered"
)

# --- CSS PERSONALIZZATO PER ESTETICA ---
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .win { color: #2ecc71; }
    .loss { color: #e74c3c; }
    div[data-testid="stMetricValue"] { font-size: 3rem; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. INPUT SIDEBAR (Impostazioni)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Impostazioni")
    
    # 1. URL GitHub
    default_url = "https://raw.githubusercontent.com/TUO_USER/TUO_REPO/main/my_bankroll.csv"
    github_url = st.text_input("URL File Bankroll (CSV Raw)", value=default_url)
    
    st.markdown("---")
    
    # 2. Parametri Kelly
    st.subheader("Parametri Kelly")
    kelly_fraction = st.slider("Kelly Fraction (Prudenza)", 0.1, 1.0, 0.25, 0.05)
    max_stake_pct = st.slider("Max Stake (% Cassa)", 0.01, 0.20, 0.05, 0.01)
    min_value = st.number_input("Valore Minimo (Edge)", value=1.05, step=0.01)

# ==============================================================================
# 3. FUNZIONI LOGICHE
# ==============================================================================
@st.cache_data(ttl=60) # Aggiorna la cache ogni 60 secondi
def load_bankroll(url):
    try:
        # Simulazione per test se l'URL non è valido
        if "TUO_USER" in url:
            return pd.DataFrame({'date': ['2024-01-01'], 'current_bankroll': [1000.0]})
            
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Errore lettura GitHub: {e}")
        return None

def calcola_kelly(prob_vittoria, quota, bankroll, fraction, max_pct, min_val):
    prob_sconfitta = 1 - prob_vittoria
    valore_atteso = (prob_vittoria * quota) - 1
    
    if valore_atteso <= (min_val - 1):
        return 0, 0, valore_atteso # Stake 0, Pct 0, Edge

    # Formula Kelly: (bp - q) / b
    b = quota - 1
    f = (b * prob_vittoria - prob_sconfitta) / b
    
    stake_pct = max(0, min(f * fraction, max_pct))
    stake_euro = bankroll * stake_pct
    
    return stake_euro, stake_pct, valore_atteso

# ==============================================================================
# 4. INTERFACCIA PRINCIPALE
# ==============================================================================
st.title("🤖 Tennis AI Betting Assistant")
st.markdown("### *Il tuo oracolo personale basato sul criterio di Kelly*")

# --- SEZIONE A: BANKROLL ---
st.markdown("---")
col1, col2 = st.columns([2, 1])

df_bankroll = load_bankroll(github_url)

if df_bankroll is not None:
    current_cassa = float(df_bankroll.iloc[-1]['current_bankroll'])
    last_update = df_bankroll.iloc[-1]['date']
    
    with col1:
        st.metric(label="💰 Bankroll Attuale", value=f"{current_cassa:.2f} €", delta=f"Aggiornato al: {last_update}")
    
    with col2:
        if st.button("🔄 Aggiorna Dati"):
            st.cache_data.clear()
            st.rerun()
else:
    st.warning("Impossibile caricare il Bankroll. Verifica l'URL nelle impostazioni.")
    current_cassa = 1000.0

# --- SEZIONE B: CALCOLATORE PUNTATA ---
st.markdown("---")
st.header("🎾 Analisi Nuova Scommessa")

with st.container():
    c1, c2, c3 = st.columns(3)
    
    with c1:
        player_name = st.text_input("Giocatore", "Jannik Sinner")
    with c2:
        bookie_odds = st.number_input("Quota Bookmaker", value=2.00, step=0.01, format="%.2f")
    with c3:
        ai_prob_pct = st.number_input("Probabilità AI (%)", value=55.0, step=0.1, format="%.1f")

    # Calcolo Live
    ai_prob = ai_prob_pct / 100.0
    stake, stake_pct, edge = calcola_kelly(ai_prob, bookie_odds, current_cassa, kelly_fraction, max_stake_pct, min_value)

    # --- OUTPUT VISIVO ---
    st.markdown("### 📊 Risultato Analisi")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric("Valore Atteso (Edge)", f"{edge*100:.2f}%", delta_color="normal")
    
    with col_res2:
        # Colore dinamico per lo stake
        color = "green" if stake > 0 else "red"
        st.markdown(f"""
            <div style="text-align: center;">
                <p style="margin-bottom: 0px;">Puntata Consigliata</p>
                <h1 style="color: {color}; margin-top: 0px;">{stake:.2f} €</h1>
            </div>
        """, unsafe_allow_html=True)
        
    with col_res3:
        st.metric("% Bankroll", f"{stake_pct*100:.2f}%")

    # Messaggio Finale
    if stake > 0:
        st.success(f"✅ **LUCE VERDE!** C'è valore su **{player_name}**. Punta **{stake:.2f}€**.")
        with st.expander("ℹ️ Spiegazione Matematica"):
            st.write(f"Secondo l'AI, {player_name} ha il **{ai_prob_pct}%** di chance.")
            st.write(f"Il Bookmaker paga come se ne avesse il **{100/bookie_odds:.1f}%**.")
            st.write(f"La differenza è il tuo vantaggio (**Edge**). Kelly suggerisce di investire il {stake_pct*100:.2f}% della cassa.")
    else:
        st.error(f"🛑 **LUCE ROSSA.** Non puntare su {player_name}.")
        st.info("Il vantaggio matematico è troppo basso o negativo. Risparmia i soldi per la prossima partita.")

# --- FOOTER ---
st.markdown("---")
st.caption("Sistema basato su XGBoost + Kelly Criterion Strategy | Dati Bankroll da GitHub")
