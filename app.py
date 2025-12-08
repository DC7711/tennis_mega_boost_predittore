import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import defaultdict
from sklearn.calibration import CalibratedClassifierCV

# =========================================================================
# CONFIGURAZIONE PAGINA E COSTANTI UTENTE
# =========================================================================
st.set_page_config(page_title="Tennis AI v6.2 + Kelly", page_icon="🎾", layout="wide")

# --- I TUOI PARAMETRI SPECIFICI ---
CURRENT_PREDICTION_DATE = pd.to_datetime("2025-12-08") 
K_BASE = 10 
MIN_MATCHES_PRIOR = 5
MIN_MATCHES_STATS = 10 
PRIOR_WEIGHT = 0.51  
PRIOR_WINDOW_MONTHS = 18 
MIN_RANK_SAFETY = 1800.0 

TOURNEY_WEIGHTS = {'G': 3.0, 'M': 2.0, 'A': 0.4, '500': 1.8, '250': 1.4, 'C': 0.7, 'F': 0.5, 'D': 1.3}
TOURNEY_NUMERIC_MAP = {'G': 4.0, 'M': 3.0, '500': 2.0, 'A': 0.3, '250': 1.0, 'D': 1.0, 'C': 0.5, 'F': 0.2}

# =========================================================================
# 1. SIDEBAR: IMPOSTAZIONI KELLY & BANKROLL (INPUT MANUALE)
# =========================================================================
with st.sidebar:
    st.header("🏦 Gestione Bankroll")
    
    # INPUT MANUALE DEL BANKROLL
    cassa_attuale = st.number_input("Il tuo Bankroll Attuale (€)", min_value=0.0, value=1000.0, step=10.0, format="%.2f")
    
    st.markdown("---")
    st.subheader("⚙️ Parametri Rischio")
    kelly_fraction = st.slider("Frazione Kelly", 0.1, 1.0, 0.25, 0.05, help="Quanto aggressivo vuoi essere? 0.25 è prudente.")
    max_stake_pct = st.slider("Max Stake (%)", 0.01, 0.20, 0.05, 0.01, help="Non puntare mai più di questa % del bankroll.")
    min_edge = st.number_input("Edge Minimo", 1.01, 1.20, 1.05, help="Scommetti solo se il valore atteso supera questa soglia.")

# =========================================================================
# FUNZIONI DI UTILITY
# =========================================================================
def key(name):
    if pd.isna(name): return ""
    name = str(name).lower().replace(".", "").strip().replace('-', ' ')
    return " ".join(name.split())

def exp(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

def get_avg_stats(pid, d, default):
    if len(d[pid]) == 0: return default
    return np.mean(d[pid][-MIN_MATCHES_STATS:])

# =========================================================================
# LOGICA CORE (CACHEATA PER VELOCITÀ)
# =========================================================================
@st.cache_resource
def load_and_train():
    status = st.empty()
    status.info("⏳ Caricamento database e training modello... (richiede circa 30-60 secondi)")
    
    # 1. CARICAMENTO DATI
    BASE_URL = 'https://raw.githubusercontent.com/Tennismylife/TML-Database/master/'
    YEARS_TO_LOAD = list(range(2019, 2026))
    df_stats_list = []
    
    STATS_COLS_TO_KEEP = [
        'tourney_date', 'winner_name', 'loser_name', 'surface',
        'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
        'w_bpSaved', 'w_bpFaced', 'l_bpSaved', 'l_bpFaced',
        'winner_hand', 'loser_hand', 'winner_age', 'loser_age',
        'winner_rank', 'loser_rank',
        'tourney_level', 'indoor', 'round', 'best_of', 'minutes'
    ]

    for year in YEARS_TO_LOAD:
        try:
            d = pd.read_csv(f"{BASE_URL}{year}.csv", low_memory=False)
            d.columns = [c.strip() for c in d.columns]
            cols = d.columns.intersection(STATS_COLS_TO_KEEP).tolist()
            df_stats_list.append(d[cols])
        except: pass

    df_main = pd.concat(df_stats_list, ignore_index=True)
    df_main['tourney_date'] = pd.to_datetime(df_main['tourney_date'], format='%Y%m%d', errors='coerce')
    df_main["w_key"] = df_main["winner_name"].apply(key)
    df_main["l_key"] = df_main["loser_name"].apply(key)

    critical_cols = ['tourney_date', 'surface', 'winner_rank', 'loser_rank', 'w_key', 'l_key']
    df = df_main.dropna(subset=critical_cols).sort_values("tourney_date").reset_index(drop=True).copy()
    df['surface'] = df['surface'].astype('category')

    # 2. FEATURE ENGINEERING
    elo = defaultdict(lambda: 1500)
    elo_surf = {"Hard": defaultdict(lambda: 1500), "Clay": defaultdict(lambda: 1500), "Grass": defaultdict(lambda: 1500)}
    
    # Dizionari Stats
    stats_dicts = {
        '1st_in': defaultdict(list), '1st_won': defaultdict(list), '2nd_won': defaultdict(list),
        'return': defaultdict(list), 'bp_save': defaultdict(list), 'bp_conv': defaultdict(list),
        'minutes': defaultdict(list)
    }

    rows = []
    np.random.seed(42)
    swap = np.random.rand(len(df)) > 0.5

    def extract_stats_match(r, prefix):
        svpt = r.get(f'{prefix}_svpt', np.nan)
        if pd.isna(svpt) or svpt == 0: return 0.60, 0.70, 0.50, 0.38, 0.60
        pct_1in = r.get(f'{prefix}_1stIn', 0) / svpt
        first_in = r.get(f'{prefix}_1stIn', 1)
        pct_1w = r.get(f'{prefix}_1stWon', 0) / first_in if first_in > 0 else 0.70
        second_serve = svpt - first_in
        pct_2w = r.get(f'{prefix}_2ndWon', 0) / second_serve if second_serve > 0 else 0.50
        bp_faced = r.get(f'{prefix}_bpFaced', 1)
        pct_bp_save = r.get(f'{prefix}_bpSaved', 0) / bp_faced if bp_faced > 0 else 0.60
        return pct_1in, pct_1w, pct_2w, 0.38, pct_bp_save

    for i, r in df.iterrows():
        w, l = r.w_key, r.l_key
        surf = r.surface if r.surface in ["Hard", "Clay", "Grass"] else "Hard"
        
        # Stats Pre-Match
        ew, el = elo.get(w, 1500), elo.get(l, 1500)
        esw, esl = elo_surf[surf].get(w, 1500), elo_surf[surf].get(l, 1500)
        
        w_stats = [get_avg_stats(w, stats_dicts[k], v) for k,v in zip(stats_dicts.keys(), [0.6,0.7,0.5,0.38,0.6,0.4,90])]
        l_stats = [get_avg_stats(l, stats_dicts[k], v) for k,v in zip(stats_dicts.keys(), [0.6,0.7,0.5,0.38,0.6,0.4,90])]

        p1_win = not swap[i]
        
        if p1_win:
            p1k, p2k = w, l
            p1r, p2r = r.winner_rank, r.loser_rank
            p1a, p2a = r.winner_age, r.loser_age
            p1h, p2h = r.winner_hand, r.loser_hand
            p1elo, p2elo = ew, el
            p1selo, p2selo = esw, esl
            s1, s2 = w_stats, l_stats
        else:
            p1k, p2k = l, w
            p1r, p2r = r.loser_rank, r.winner_rank
            p1a, p2a = r.loser_age, r.winner_age
            p1h, p2h = r.loser_hand, r.winner_hand
            p1elo, p2elo = el, ew
            p1selo, p2selo = esl, esw
            s1, s2 = l_stats, w_stats

        p1r = np.clip(p1r, 1.0, MIN_RANK_SAFETY) if pd.notna(p1r) else MIN_RANK_SAFETY
        p2r = np.clip(p2r, 1.0, MIN_RANK_SAFETY) if pd.notna(p2r) else MIN_RANK_SAFETY
        p1a = p1a if pd.notna(p1a) else 25.0
        p2a = p2a if pd.notna(p2a) else 25.0

        lvl_code = str(r.get('tourney_level')).strip()
        lvl_num = TOURNEY_NUMERIC_MAP.get(lvl_code, 1.0)

        row = {
            "tourney_date": r.tourney_date,
            "target": 1 if p1_win else 0,
            "surface": surf,
            "log_rank_diff": np.log(p2r) - np.log(p1r),
            "elo_diff": p1elo - p2elo,
            "surf_elo_diff": p1selo - p2selo,
            "1st_in_diff": s1[0] - s2[0],
            "1st_won_diff": s1[1] - s2[1],
            "2nd_won_diff": s1[2] - s2[2],
            "return_diff": s1[3] - s2[3],
            "bp_save_diff": s1[4] - s2[4],
            "bp_conv_diff": s1[5] - s2[5],
            "p1_age": p1a, "p2_age": p2a,
            "p1_is_left": 1 if p1h == 'L' else 0, "p2_is_left": 1 if p2h == 'L' else 0,
            "is_indoor": 1 if r.get('indoor', 'O') == 'I' else 0,
            "is_best_of_5": 1 if r.get('best_of', 3) == 5 else 0,
            "avg_minutes_diff": s1[6] - s2[6],
            "tourney_numeric_level": lvl_num
        }
        rows.append(row)

        # Aggiornamento Post-Match
        Kw = K_BASE * TOURNEY_WEIGHTS.get(lvl_code, 1.0)
        ex, sx = exp(ew, el), exp(esw, esl)
        elo[w] += Kw * (1-ex); elo[l] -= Kw * ex
        elo_surf[surf][w] += Kw * (1-sx); elo_surf[surf][l] -= Kw * sx
        
        ws = extract_stats_match(r, 'w')
        ls = extract_stats_match(r, 'l')
        
        l_bpf = r.get('l_bpFaced', 1)
        w_conv = (l_bpf - r.get('l_bpSaved',0))/l_bpf if l_bpf>0 else 0.40
        w_bpf = r.get('w_bpFaced', 1)
        l_conv = (w_bpf - r.get('w_bpSaved',0))/w_bpf if w_bpf>0 else 0.40
        
        # Append stats
        stats_dicts['1st_in'][w].append(ws[0]); stats_dicts['1st_in'][l].append(ls[0])
        stats_dicts['1st_won'][w].append(ws[1]); stats_dicts['1st_won'][l].append(ls[1])
        stats_dicts['2nd_won'][w].append(ws[2]); stats_dicts['2nd_won'][l].append(ls[2])
        
        w_ret = 1 - (ls[1]*ls[0] + ls[2]*(1-ls[0]))
        l_ret = 1 - (ws[1]*ws[0] + ws[2]*(1-ws[0]))
        stats_dicts['return'][w].append(w_ret); stats_dicts['return'][l].append(l_ret)
        
        stats_dicts['bp_save'][w].append(ws[4]); stats_dicts['bp_save'][l].append(ls[4])
        stats_dicts['bp_conv'][w].append(w_conv); stats_dicts['bp_conv'][l].append(l_conv)
        
        m_min = r.get('minutes', np.nan)
        if pd.notna(m_min) and m_min > 0:
            stats_dicts['minutes'][w].append(m_min); stats_dicts['minutes'][l].append(m_min)

    ML = pd.DataFrame(rows).fillna(0)
    ML = pd.get_dummies(ML, columns=['surface'], prefix='surface')
    
    # 3. TRAINING
    features = [c for c in ML.columns if c not in ['tourney_date', 'target']]
    split_date = pd.to_datetime("2025-09-01")
    mask = ML["tourney_date"] < split_date
    
    X_train, y_train = ML.loc[mask, features], ML.loc[mask, "target"]
    X_test, y_test = ML.loc[~mask, features], ML.loc[~mask, "target"]
    
    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.01, max_depth=5, 
        subsample=0.7, colsample_bytree=0.7, eval_metric="logloss",
        random_state=42, early_stopping_rounds=70
    )
    
    if not X_test.empty:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv="prefit")
        calibrator.fit(X_test, y_test)
        predictor = calibrator
    else:
        model.fit(X_train, y_train)
        predictor = model

    status.success("✅ Modello Allenato con Successo!")
    
    # Return everything needed for prediction
    return predictor, df, elo, elo_surf, stats_dicts, features

# Caricamento Modello
predictor, df, elo, elo_surf, stats_dicts, features = load_and_train()

# =========================================================================
# UI INTERFACCIA
# =========================================================================
st.title("🎾 Tennis AI v6.2 - Web App")
st.markdown(f"**Current Date:** {CURRENT_PREDICTION_DATE.date()} | **Prior Weight:** {PRIOR_WEIGHT} | **Prior Multiplier:** 0.8")

col1, col2 = st.columns(2)
with col1:
    all_players = sorted(list(set(df['winner_name'].unique()) | set(df['loser_name'].unique())))
    p1_name = st.selectbox("Giocatore 1", all_players, index=all_players.index("Jannik Sinner") if "Jannik Sinner" in all_players else 0)
    p2_name = st.selectbox("Giocatore 2", all_players, index=all_players.index("Carlos Alcaraz") if "Carlos Alcaraz" in all_players else 0)

with col2:
    surface = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    level = st.selectbox("Livello Torneo", ["G", "M", "500", "250", "A", "C", "F"])
    is_indoor = st.checkbox("Indoor")
    is_bo5 = st.checkbox("Best of 5 Sets", value=(level=="G"))

# =========================================================================
# LOGICA PREDIZIONE
# =========================================================================
def get_smart_prior_components(player_key, surface_key, is_bo5, data_df_prior):
    recent_matches = data_df_prior[
        ((data_df_prior['w_key'] == player_key) | (data_df_prior['l_key'] == player_key)) &
        (data_df_prior['surface'] == surface_key)
    ]
    W_surf = (recent_matches['w_key'] == player_key).sum()
    L_surf = (recent_matches['l_key'] == player_key).sum()
    N_surf = W_surf + L_surf
    
    if N_surf < MIN_MATCHES_PRIOR:
        all_matches = data_df_prior[((data_df_prior['w_key'] == player_key) | (data_df_prior['l_key'] == player_key))]
        W_tot = (all_matches['w_key'] == player_key).sum()
        L_tot = (all_matches['l_key'] == player_key).sum()
        W = W_surf * 0.6 + W_tot * 0.4
        L = L_surf * 0.6 + L_tot * 0.4
        N = N_surf * 0.6 + L_tot * 0.4
    else:
        W, L, N = W_surf, L_surf, N_surf

    if N < MIN_MATCHES_PRIOR * 0.8: prob = 0.5
    else: prob = (W + 1) / (N + 2)
    
    if is_bo5 and prob > 0.55: prob = np.clip(prob + 0.01, 0.05, 0.95)
    return prob

if st.button("🔮 PREDICI MATCH"):
    k1, k2 = key(p1_name), key(p2_name)
    surf_key = surface if surface in ["Hard", "Clay", "Grass"] else "Hard"
    
    # 1. Recupero Dati
    def get_latest(k):
        ms = df[(df.w_key==k)|(df.l_key==k)]
        if ms.empty: return MIN_RANK_SAFETY, 25.0, 'R'
        last = ms.sort_values('tourney_date').iloc[-1]
        win = last.w_key == k
        return (last.winner_rank if win else last.loser_rank), (last.winner_age if win else last.loser_age), (last.winner_hand if win else last.loser_hand)

    p1r, p1a, p1h = get_latest(k1)
    p2r, p2a, p2h = get_latest(k2)
    p1r = np.clip(p1r, 1, MIN_RANK_SAFETY); p2r = np.clip(p2r, 1, MIN_RANK_SAFETY)
    
    # 2. ML Prediction
    stats_k = ['1st_in', '1st_won', '2nd_won', 'return', 'bp_save', 'bp_conv', 'minutes']
    defaults = [0.6, 0.7, 0.5, 0.38, 0.6, 0.4, 90]
    
    s1 = [get_avg_stats(k1, stats_dicts[k], v) for k,v in zip(stats_k, defaults)]
    s2 = [get_avg_stats(k2, stats_dicts[k], v) for k,v in zip(stats_k, defaults)]
    
    row = {
        "log_rank_diff": np.log(p2r) - np.log(p1r),
        "elo_diff": elo[k1] - elo[k2],
        "surf_elo_diff": elo_surf[surf_key][k1] - elo_surf[surf_key][k2],
        "1st_in_diff": s1[0] - s2[0], "1st_won_diff": s1[1] - s2[1], "2nd_won_diff": s1[2] - s2[2],
        "return_diff": s1[3] - s2[3], "bp_save_diff": s1[4] - s2[4], "bp_conv_diff": s1[5] - s2[5],
        "p1_age": p1a, "p2_age": p2a,
        "p1_is_left": 1 if p1h == 'L' else 0, "p2_is_left": 1 if p2h == 'L' else 0,
        "is_indoor": 1 if is_indoor else 0, "is_best_of_5": 1 if is_bo5 else 0,
        "avg_minutes_diff": s1[6] - s2[6],
        "tourney_numeric_level": TOURNEY_NUMERIC_MAP.get(level, 1.0)
    }
    for s in ['Hard', 'Clay', 'Grass', 'Carpet']: row[f'surface_{s}'] = 1 if s == surf_key else 0
    
    df_in = pd.DataFrame([row])
    for c in features: 
        if c not in df_in.columns: df_in[c] = 0
            
    prob_ml = predictor.predict_proba(df_in[features])[0][1]
    
    # 3. Prior
    start_prior = CURRENT_PREDICTION_DATE - pd.DateOffset(months=PRIOR_WINDOW_MONTHS)
    df_prior = df[df['tourney_date'] >= start_prior].copy()
    
    p1_w_pct = get_smart_prior_components(k1, surf_key, is_bo5, df_prior)
    p2_w_pct = get_smart_prior_components(k2, surf_key, is_bo5, df_prior)
    
    w_l_prior = 0.5 + (p1_w_pct - p2_w_pct) * 0.8  
    w_l_prior = np.clip(w_l_prior, 0.05, 0.95)
    
    prob_elo_base = exp(elo[k2], elo[k1])
    prob_surf_base = exp(elo_surf[surf_key][k2], elo_surf[surf_key][k1])
    
    prob_prior = (w_l_prior * 0.90) + (prob_surf_base * 0.08) + (prob_elo_base * 0.02)
    prob_prior = np.clip(prob_prior, 0.05, 0.95)
    
    # 4. Final
    final_prob = (prob_ml * (1 - PRIOR_WEIGHT)) + (prob_prior * PRIOR_WEIGHT)
    
    # =========================================================================
    # 5. VISUALIZZAZIONE RISULTATO AI
    # =========================================================================
    st.divider()
    
    # Determiniamo chi è il "Prescelto" dal sistema
    if prob_p1_final > 0.5:
        system_pick = p1_name
        system_prob = prob_p1_final
        system_odds = quota_p1
        loser_pick = p2_name
        loser_prob = prob_p2_final
    else:
        system_pick = p2_name
        system_prob = prob_p2_final
        system_odds = quota_p2
        loser_pick = p1_name
        loser_prob = prob_p1_final

    st.subheader(f"🧠 Il Sistema sceglie: {system_pick}")
    
    # Barra grafica
    st.progress(prob_p1_final)
    c1, c2 = st.columns([1,1])
    c1.caption(f"{p1_name}: {prob_p1_final:.1%}")
    c2.caption(f"{p2_name}: {prob_p2_final:.1%}")

    # =========================================================================
    # 6. MODULO KELLY DIREZIONALE (SOLO SUL VINCENTE)
    # =========================================================================
    st.markdown("---")
    st.header("💰 Calcolo Puntata (Kelly)")

    # 1. Calcolo Edge SOLO sulla scelta del sistema
    edge = (system_prob * system_odds) - 1
    
    # La soglia min_edge nella sidebar è tipo 1.05 (che significa +5% edge)
    # Quindi l'edge matematico deve essere > 0.05
    min_edge_val = min_edge - 1.0 

    col_k1, col_k2 = st.columns(2)
    
    with col_k1:
        st.metric("Tuo Bankroll", f"{cassa_attuale:.2f} €")
        st.write(f"Probabilità AI: **{system_prob:.1%}**")
        st.write(f"Quota Bookmaker: **{system_odds:.2f}**")

    with col_k2:
        # LOGICA RIGIDA: Si gioca solo se Edge è positivo E superiore al minimo
        if edge >= min_edge_val:
            # Formula Kelly
            b = system_odds - 1
            q = 1 - system_prob
            f = (b * system_prob - q) / b
            
            # Applicazione Frazione e Limiti
            stake_pct = max(0, min(f * kelly_fraction, max_stake_pct))
            stake_euro = cassa_attuale * stake_pct
            
            st.success(f"✅ **PUNTARE SU {system_pick.upper()}**")
            st.metric("Importo Puntata", f"{stake_euro:.2f} €")
            st.caption(f"ROI Atteso: +{edge*100:.2f}% | Stake: {stake_pct*100:.2f}% del roll")
        
        else:
            # Caso No Bet
            st.error("🛑 **NESSUNA PUNTATA (No Value)**")
            
            if edge < 0:
                st.write(f"Motivo: La quota {system_odds} non copre il rischio (Prob {system_prob:.1%}).")
            else:
                st.write(f"Motivo: Edge ({edge*100:.2f}%) inferiore al minimo richiesto ({(min_edge-1)*100:.0f}%).")
            
            required_odds = (1 + min_edge_val) / system_prob
            st.info(f"💡 Per puntare su {system_pick} servirebbe una quota minima di **{required_odds:.2f}**")
