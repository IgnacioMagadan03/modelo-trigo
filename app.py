# =============================================================
# app.py - Modelo híbrido ARIMA + XGBoost para trigo AR
# =============================================================
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

import streamlit as st

BASE = Path(".")

# ---------------------------------------------------------
# 1. FUNCIÓN PRINCIPAL: entrena el modelo y devuelve
#    - data_hist: histórico en nivel
#    - forecast_df: pronóstico 12 meses
#    - mae_test, r2_test: métricas de los últimos 12 meses
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def entrenar_modelo():
    # =========================================================
    # 1) SERIE PRINCIPAL ARGENTINA
    # =========================================================
    trigo = pd.read_excel(BASE / "Trigo_Ajustado_MENSUAL_ONLY.xlsx")
    trigo['PeriodoYM'] = pd.to_datetime(trigo['PeriodoYM'])
    trigo = trigo.sort_values('PeriodoYM').reset_index(drop=True)
    trigo['Mes'] = trigo['PeriodoYM'].dt.month

    # =========================================================
    # 2) FUTUROS USA
    # =========================================================
    fut = pd.read_csv(BASE / "Datos históricos Futuros trigo EE.UU..csv")
    fut['Fecha'] = pd.to_datetime(fut['Fecha'], format="%d.%m.%Y")

    fut['Cierre'] = (
        fut['Cierre'].astype(str)
                     .str.strip()
                     .str.replace(',', '', regex=False)
    )
    fut['Cierre'] = pd.to_numeric(fut['Cierre'], errors='coerce')
    fut['PeriodoYM'] = fut['Fecha'].values.astype("datetime64[M]")

    fut_m = (
        fut.sort_values('Fecha')
           .groupby('PeriodoYM')['Cierre']
           .last()
           .reset_index()
           .rename(columns={'Cierre': 'Futuro_USA'})
    )

    # =========================================================
    # 3) USD/ARS
    # =========================================================
    usd = pd.read_csv(BASE / "Datos históricos USD_ARS (1).csv")
    usd['Fecha'] = pd.to_datetime(usd['Fecha'], format="%d.%m.%Y")

    usd['Cierre'] = (
        usd['Cierre'].astype(str)
                     .str.strip()
                     .str.replace(',', '', regex=False)
    )
    usd['Cierre'] = pd.to_numeric(usd['Cierre'], errors='coerce')
    usd['PeriodoYM'] = usd['Fecha'].values.astype("datetime64[M]")

    usd_m = (
        usd.sort_values('Fecha')
           .groupby('PeriodoYM')['Cierre']
           .last()
           .reset_index()
           .rename(columns={'Cierre': 'USD_ARS'})
    )

    # =========================================================
    # 4) MERGE BÁSICO + FEATURES
    # =========================================================
    data = (
        trigo
          .merge(fut_m, on='PeriodoYM', how='left')
          .merge(usd_m, on='PeriodoYM', how='left')
    )

    data = data.sort_values('PeriodoYM').reset_index(drop=True)

    # Tendencia temporal
    data['t'] = np.arange(len(data))

    # Trabajamos en logaritmos
    data = data.dropna(subset=['Precio_Hoy', 'Futuro_USA', 'USD_ARS']).copy()
    data['log_precio'] = np.log(data['Precio_Hoy'])
    data['log_fut']    = np.log(data['Futuro_USA'])
    data['log_usd']    = np.log(data['USD_ARS'])

    # Índice por fecha para ARIMA
    data = data.set_index('PeriodoYM')

    # =========================================================
    # 5) ARIMA SOLO SOBRE log_precio (SIN estacionalidad explícita)
    # =========================================================
    y = data['log_precio']

    arima = SARIMAX(
        y,
        order=(1, 1, 1),           # sin seasonal_order
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_arima = arima.fit(disp=False)

    # Predicción (one step ahead approx.) y residuos
    arima_pred = res_arima.get_prediction().predicted_mean
    residuals = y - arima_pred
    data['resid_arima'] = residuals

    # =========================================================
    # 6) XGBOOST SOBRE LOS RESIDUOS
    # =========================================================
    df = data.copy()

    # Lags del residuo
    for lag in range(1, 7):
        df[f'resid_lag{lag}'] = df['resid_arima'].shift(lag)

    # Mes como dummies
    df['Mes'] = df.index.month
    df = pd.get_dummies(df, columns=['Mes'], drop_first=True)

    # Filas completas
    df_model = df.dropna(subset=['resid_arima', 'log_fut', 'log_usd'])

    feature_cols = ['log_fut', 'log_usd', 't'] + [f'resid_lag{lag}' for lag in range(1, 7)]
    feature_cols += [c for c in df_model.columns if c.startswith('Mes_')]

    X_res = df_model[feature_cols]
    y_res = df_model['resid_arima']

    # =========================================================
    # 7) TRAIN / TEST (últimos 12 meses para evaluar)
    # =========================================================
    test_horizon = 12
    X_res_train, X_res_test = X_res.iloc[:-test_horizon], X_res.iloc[-test_horizon:]
    y_res_train, y_res_test = y_res.iloc[:-test_horizon], y_res.iloc[-test_horizon:]

    # Alineamos pred ARIMA con df_model
    arima_aligned = arima_pred.reindex(df_model.index)
    arima_test = arima_aligned.iloc[-test_horizon:]

    # =========================================================
    # 8) XGBOOST
    # =========================================================
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=0
    )
    xgb.fit(X_res_train, y_res_train)

    res_test_pred = xgb.predict(X_res_test)

    # Pred final en log-precio = ARIMA + residuo ML
    log_pred_final_test = arima_test + res_test_pred
    log_real_test = y.reindex(df_model.index).iloc[-test_horizon:]

    precio_pred_test = np.exp(log_pred_final_test)
    precio_real_test = np.exp(log_real_test)

    mae_test = mean_absolute_error(precio_real_test, precio_pred_test)
    r2_test = r2_score(log_real_test, log_pred_final_test)

    # =========================================================
    # 9) PRONÓSTICO 12 MESES ADELANTE
    # =========================================================
    steps_ahead = 12
    forecast_arima = res_arima.get_forecast(steps=steps_ahead)
    log_arima_future_raw = forecast_arima.predicted_mean

    # Fechas futuras
    last_date = df_model.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=steps_ahead,
        freq='MS'
    )

    # Reindexamos el forecast ARIMA a esas fechas
    log_arima_future = pd.Series(log_arima_future_raw.values, index=future_dates)

    # Forecast residuos con XGBoost (escenario estable de Chicago y USD)
    last_row = df_model.iloc[-1]
    res_lags = [last_row[f'resid_lag{lag}'] for lag in range(1, 7)]
    future_resids = []

    for step, fecha in enumerate(future_dates, start=1):
        mes = fecha.month

        feat = {
            'log_fut': last_row['log_fut'],   # escenario estable
            'log_usd': last_row['log_usd'],
            't': last_row['t'] + step,
        }

        # lags
        for i, val in enumerate(res_lags, start=1):
            feat[f'resid_lag{i}'] = val

        # dummies de mes
        for c in [c for c in df_model.columns if c.startswith('Mes_')]:
            feat[c] = 0
        if f'Mes_{mes}' in feat:
            feat[f'Mes_{mes}'] = 1

        X_fut = pd.DataFrame([feat])[feature_cols]
        pred_res = xgb.predict(X_fut)[0]
        future_resids.append(pred_res)

        # actualizamos lags con el nuevo residuo
        res_lags = [pred_res] + res_lags[:-1]

    log_resid_future = pd.Series(future_resids, index=future_dates)

    # Pred final future = ARIMA future + residuo future
    log_precio_future = log_arima_future + log_resid_future
    precio_future = np.exp(log_precio_future)

    forecast_df = pd.DataFrame({
        'PeriodoYM': future_dates,
        'Precio_Pronosticado': precio_future.values
    })

    # Data histórica en nivel
    data_hist = data.copy().reset_index()
    data_hist['Precio_Hoy'] = np.exp(data_hist['log_precio'])

    return data_hist, forecast_df, mae_test, r2_test


# =============================================================
# 2. INTERFAZ STREAMLIT
# =============================================================
st.set_page_config(
    page_title="Modelo trigo AR",
    layout="wide"
)

st.title("🌾 Pronóstico del precio del trigo en Argentina")
st.caption("Modelo híbrido ARIMA (sin estacionalidad explícita) + XGBoost sobre residuos")

with st.spinner("Entrenando modelo y calculando pronósticos..."):
    data_hist, forecast_df, mae_test, r2_test = entrenar_modelo()

# ---------------- KPIs ----------------
col1, col2 = st.columns(2)
with col1:
    st.metric("MAE (últimos 12 meses)", f"{mae_test:,.0f} ARS/tn")
with col2:
    st.metric("R² (log-precio)", f"{r2_test:.3f}")

st.markdown("---")

# ---------------- TABS: GRÁFICO Y TABLA ----------------
tab1, tab2 = st.tabs(["📈 Histórico + pronóstico", "📊 Tabla de pronósticos"])

with tab1:
    st.subheader("Evolución del precio y pronóstico 12 meses")

    hist_tail = data_hist.tail(36)  # últimos 3 años aprox.

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_tail["PeriodoYM"], hist_tail["Precio_Hoy"], label="Histórico")
    ax.plot(
        forecast_df["PeriodoYM"],
        forecast_df["Precio_Pronosticado"],
        marker="o",
        linestyle="--",
        label="Pronóstico 12 meses"
    )
    ax.legend()
    ax.set_ylabel("Precio (ARS constantes)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

with tab2:
    st.subheader("Detalle de pronósticos por mes")

    forecast_df_show = forecast_df.copy()
    forecast_df_show["PeriodoYM"] = forecast_df_show["PeriodoYM"].dt.strftime("%Y-%m")
    st.dataframe(
        forecast_df_show.set_index("PeriodoYM"),
        use_container_width=True
    )

st.markdown("---")

# ---------------- CONSULTA PUNTUAL ----------------
st.subheader(" Consultar un mes puntual")

forecast_df_show = forecast_df.copy()
forecast_df_show["PeriodoYM_str"] = forecast_df_show["PeriodoYM"].dt.strftime("%Y-%m")

opcion = st.selectbox(
    "Elegí el mes de interés",
    options=forecast_df_show["PeriodoYM_str"].tolist()
)

if opcion:
    fila = forecast_df_show.loc[forecast_df_show["PeriodoYM_str"] == opcion].iloc[0]
    valor = fila["Precio_Pronosticado"]
    st.write(
        f" Pronóstico para **{opcion}**: **{valor:,.0f} ARS/tn (constantes)**"
    )
