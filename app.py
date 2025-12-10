# =============================================================
# app.py - Consola interactiva de precios de trigo (Argentina)
# Modelo híbrido: ARIMA (parte temporal) + XGBoost (corrección)
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

import streamlit as st
from pandas.tseries.offsets import MonthBegin

BASE = Path(".")

# ---------------------------------------------------------
# FUNCIÓN: Entrena el modelo y arma el escenario base
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def entrenar_modelo():
    # =========================================================
    # 1) SERIE PRINCIPAL ARGENTINA
    # =========================================================
    trigo = pd.read_excel(BASE / "Trigo_Ajustado_MENSUAL_ONLY.xlsx")
    trigo["PeriodoYM"] = pd.to_datetime(trigo["PeriodoYM"])
    trigo = trigo.sort_values("PeriodoYM").reset_index(drop=True)
    trigo["Mes"] = trigo["PeriodoYM"].dt.month

    # =========================================================
    # 2) FUTUROS USA (Chicago)
    # =========================================================
    fut = pd.read_csv(BASE / "Datos históricos Futuros trigo EE.UU..csv")
    fut["Fecha"] = pd.to_datetime(fut["Fecha"], format="%d.%m.%Y")

    fut["Cierre"] = (
        fut["Cierre"]
        .astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
    )
    fut["Cierre"] = pd.to_numeric(fut["Cierre"], errors="coerce")
    fut["PeriodoYM"] = fut["Fecha"].values.astype("datetime64[M]")

    fut_m = (
        fut.sort_values("Fecha")
        .groupby("PeriodoYM")["Cierre"]
        .last()
        .reset_index()
        .rename(columns={"Cierre": "Futuro_USA"})
    )

    # =========================================================
    # 3) USD/ARS
    # =========================================================
    usd = pd.read_csv(BASE / "Datos históricos USD_ARS (1).csv")
    usd["Fecha"] = pd.to_datetime(usd["Fecha"], format="%d.%m.%Y")

    usd["Cierre"] = (
        usd["Cierre"]
        .astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
    )
    usd["Cierre"] = pd.to_numeric(usd["Cierre"], errors="coerce")
    usd["PeriodoYM"] = usd["Fecha"].values.astype("datetime64[M]")

    usd_m = (
        usd.sort_values("Fecha")
        .groupby("PeriodoYM")["Cierre"]
        .last()
        .reset_index()
        .rename(columns={"Cierre": "USD_ARS"})
    )

    # =========================================================
    # 4) MERGE BÁSICO + FEATURES
    # =========================================================
    data = (
        trigo
        .merge(fut_m, on="PeriodoYM", how="left")
        .merge(usd_m, on="PeriodoYM", how="left")
    )

    data = data.sort_values("PeriodoYM").reset_index(drop=True)
    data["t"] = np.arange(len(data))

    data = data.dropna(subset=["Precio_Hoy", "Futuro_USA", "USD_ARS"]).copy()
    data["log_precio"] = np.log(data["Precio_Hoy"])
    data["log_fut"] = np.log(data["Futuro_USA"])
    data["log_usd"] = np.log(data["USD_ARS"])

    data = data.set_index("PeriodoYM")

    # =========================================================
    # 5) ARIMA SOBRE log_precio (SIN estacionalidad explícita)
    # =========================================================
    y = data["log_precio"]

    arima = SARIMAX(
        y,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res_arima = arima.fit(disp=False)

    arima_pred_full = res_arima.get_prediction().predicted_mean
    residuals = y - arima_pred_full
    data["resid_arima"] = residuals

    # =========================================================
    # 6) XGBOOST SOBRE RESIDUOS
    # =========================================================
    df = data.copy()

    # Lags del residuo
    for lag in range(1, 7):
        df[f"resid_lag{lag}"] = df["resid_arima"].shift(lag)

    # Mes como dummies
    df["Mes"] = df.index.month
    df = pd.get_dummies(df, columns=["Mes"], drop_first=True)

    df_model = df.dropna(subset=["resid_arima", "log_fut", "log_usd"])

    feature_cols = ["log_fut", "log_usd", "t"] + [
        f"resid_lag{lag}" for lag in range(1, 7)
    ]
    feature_cols += [c for c in df_model.columns if c.startswith("Mes_")]

    X_res = df_model[feature_cols]
    y_res = df_model["resid_arima"]

    # =========================================================
    # 7) TRAIN / TEST (últimos 12 meses)
    # =========================================================
    test_horizon = 12
    X_res_train, X_res_test = X_res.iloc[:-test_horizon], X_res.iloc[-test_horizon:]
    y_res_train, y_res_test = y_res.iloc[:-test_horizon], y_res.iloc[-test_horizon:]

    # Alineamos ARIMA con df_model
    arima_aligned = arima_pred_full.reindex(df_model.index)
    arima_test = arima_aligned.iloc[-test_horizon:]

    # Naïve (precio de ayer)
    log_real_all = y.reindex(df_model.index)
    log_real_test = log_real_all.iloc[-test_horizon:]
    log_naive_test = log_real_all.shift(1).iloc[-test_horizon:]

    # ARIMA solo (sin XGB)
    log_arima_test = arima_test

    # =========================================================
    # 8) XGBOOST
    # =========================================================
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=0,
    )
    xgb.fit(X_res_train, y_res_train)

    res_test_pred = xgb.predict(X_res_test)

    # Híbrido: ARIMA + residuo ML
    log_pred_hybrid_test = log_arima_test + res_test_pred

    # Pasamos todo a nivel
    precio_real_test = np.exp(log_real_test)
    precio_naive_test = np.exp(log_naive_test)
    precio_arima_test = np.exp(log_arima_test)
    precio_hybrid_test = np.exp(log_pred_hybrid_test)

    mae_naive = mean_absolute_error(precio_real_test, precio_naive_test)
    mae_arima = mean_absolute_error(precio_real_test, precio_arima_test)
    mae_hybrid = mean_absolute_error(precio_real_test, precio_hybrid_test)

    # =========================================================
    # 9) PRONÓSTICO 12 MESES ADELANTE (ESCENARIO BASE)
    # =========================================================
    steps_ahead = 12
    forecast_arima = res_arima.get_forecast(steps=steps_ahead)
    log_arima_future_raw = forecast_arima.predicted_mean

    last_date = df_model.index[-1]
    future_dates = pd.date_range(
        start=last_date + MonthBegin(1),
        periods=steps_ahead,
        freq="MS",
    )

    log_arima_future_base = pd.Series(log_arima_future_raw.values, index=future_dates)

    # Residuos futuros con XGB (dólar y Chicago estables)
    last_row = df_model.iloc[-1]
    res_lags = [last_row[f"resid_lag{lag}"] for lag in range(1, 7)]
    future_resids_base = []

    for step, fecha in enumerate(future_dates, start=1):
        mes = fecha.month

        feat = {
            "log_fut": last_row["log_fut"],
            "log_usd": last_row["log_usd"],
            "t": last_row["t"] + step,
        }

        for i, val in enumerate(res_lags, start=1):
            feat[f"resid_lag{i}"] = val

        for c in [c for c in df_model.columns if c.startswith("Mes_")]:
            feat[c] = 0
        if f"Mes_{mes}" in feat:
            feat[f"Mes_{mes}"] = 1

        X_fut = pd.DataFrame([feat])[feature_cols]
        pred_res = xgb.predict(X_fut)[0]
        future_resids_base.append(pred_res)

        res_lags = [pred_res] + res_lags[:-1]

    log_resid_future_base = pd.Series(future_resids_base, index=future_dates)

    log_precio_future_base = log_arima_future_base + log_resid_future_base
    precio_future_base = np.exp(log_precio_future_base)

    forecast_base_df = pd.DataFrame(
        {
            "PeriodoYM": future_dates,
            "Precio_Pronosticado": precio_future_base.values,
        }
    )

    # Data histórica en nivel
    data_hist = data.copy().reset_index()
    data_hist["Precio_Hoy"] = np.exp(data_hist["log_precio"])

    metrics = {
        "MAE_naive": mae_naive,
        "MAE_arima": mae_arima,
        "MAE_hybrid": mae_hybrid,
    }

    return {
        "data_hist": data_hist,
        "forecast_base": forecast_base_df,
        "metrics": metrics,
        "xgb": xgb,
        "feature_cols": feature_cols,
    }


# =============================================================
# INTERFAZ STREAMLIT
# =============================================================
st.set_page_config(
    page_title="Precio trigo AR - Consola interactiva",
    layout="wide"
)

st.title("🌾Precios de trigo Argentina🌾")
st.caption(

with st.spinner("Entrenando modelo y calculando escenario base..."):
    res = entrenar_modelo()

data_hist = res["data_hist"]
forecast_base = res["forecast_base"]
metrics = res["metrics"]
xgb_model = res["xgb"]
feature_cols = res["feature_cols"]

# =============================================================
# KPIs
# =============================================================
ultimo_precio = data_hist["Precio_Hoy"].iloc[-1]
prox_mes_base = forecast_base["Precio_Pronosticado"].iloc[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Último precio observado", f"{ultimo_precio:,.0f} ARS/tn")
with col2:
    st.metric("Pronóstico próximo mes (escenario base)", f"{prox_mes_base:,.0f} ARS/tn")
with col3:
    st.metric("MAE modelo híbrido (últimos 12 meses)", f"{metrics['MAE_hybrid']:,.0f} ARS/tn")

st.markdown("---")

# =============================================================
# TABS PRINCIPALES
# =============================================================
tab1, tab2, tab3 = st.tabs(
    ["📈 Escenario base", "🎛 Escenarios interactivos", "📊 Modelo y métricas"]
)

# ---------------- TAB 1: ESCENARIO BASE ----------------
with tab1:
    st.subheader("Histórico + pronóstico (escenario base)")

    hist_tail = data_hist.tail(36)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        hist_tail["PeriodoYM"],
        hist_tail["Precio_Hoy"],
        label="Histórico",
        color="#1f77b4",
    )
    ax.plot(
        forecast_base["PeriodoYM"],
        forecast_base["Precio_Pronosticado"],
        marker="o",
        linestyle="--",
        label="Pronóstico base (12 meses)",
        color="#2ca02c",
    )
    ax.legend()
    ax.set_ylabel("Precio (ARS constantes)")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

    st.subheader("Tabla de pronóstico (escenario base)")
    forecast_base_show = forecast_base.copy()
    forecast_base_show["PeriodoYM"] = forecast_base_show["PeriodoYM"].dt.strftime(
        "%Y-%m"
    )
    st.dataframe(
        forecast_base_show.set_index("PeriodoYM"),
        use_container_width=True,
    )

    csv_base = forecast_base.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar pronóstico base (CSV)",
        data=csv_base,
        file_name="pronostico_base_trigo.csv",
        mime="text/csv",
    )

    st.subheader("Consultar un mes puntual")
    opciones = forecast_base_show["PeriodoYM"].tolist()
    mes_sel = st.selectbox("Elegí un mes", opciones, key="mes_base")
    if mes_sel:
        valor = forecast_base_show.loc[
            forecast_base_show["PeriodoYM"] == mes_sel, "Precio_Pronosticado"
        ].iloc[0]
        st.write(f"Pronóstico para **{mes_sel}**: **{valor:,.0f} ARS/tn**")


# ---------------- TAB 2: ESCENARIOS INTERACTIVOS ----------------
with tab2:
    st.subheader("Escenarios interactivos: Dólar y Chicago")
    st.markdown(
        """
Estos escenarios NO recalculan todo el modelo desde cero, sino que aplican
un **ajuste multiplicativo** sobre el pronóstico base, asumiendo que cambios
en dólar y Chicago se trasladan parcialmente al precio interno.
"""
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Escenario A")
        usd_A = st.slider(
            "Ajuste USD/ARS (%) - Escenario A",
            -30,
            30,
            0,
            step=5,
            help="Sube o baja el tipo de cambio respecto al último valor",
            key="usd_A",
        )
        fut_A = st.slider(
            "Ajuste Futuro Chicago (%) - Escenario A",
            -30,
            30,
            0,
            step=5,
            help="Sube o baja el futuro de trigo en Chicago",
            key="fut_A",
        )

    with colB:
        st.markdown("#### Escenario B")
        usd_B = st.slider(
            "Ajuste USD/ARS (%) - Escenario B",
            -30,
            30,
            0,
            step=5,
            help="Sube o baja el tipo de cambio respecto al último valor",
            key="usd_B",
        )
        fut_B = st.slider(
            "Ajuste Futuro Chicago (%) - Escenario B",
            -30,
            30,
            0,
            step=5,
            help="Sube o baja el futuro de trigo en Chicago",
            key="fut_B",
        )

    # Sensibilidades asumidas (puedes ajustarlas)
    alpha_usd = 0.5  # 50% del cambio del dólar se traslada al precio
    alpha_fut = 0.3  # 30% del cambio de Chicago se traslada al precio

    factor_A = 1 + (alpha_usd * usd_A + alpha_fut * fut_A) / 100
    factor_B = 1 + (alpha_usd * usd_B + alpha_fut * fut_B) / 100

    esc_A = forecast_base.copy()
    esc_B = forecast_base.copy()
    esc_A["Precio_Pronosticado"] = esc_A["Precio_Pronosticado"] * factor_A
    esc_B["Precio_Pronosticado"] = esc_B["Precio_Pronosticado"] * factor_B

    # Gráfico comparación
    fig2, ax2 = plt.subplots(figsize=(10, 4))

    hist_tail = data_hist.tail(36)
    ax2.plot(
        hist_tail["PeriodoYM"],
        hist_tail["Precio_Hoy"],
        label="Histórico",
        color="#1f77b4",
    )

    ax2.plot(
        forecast_base["PeriodoYM"],
        forecast_base["Precio_Pronosticado"],
        linestyle="--",
        label="Base",
        color="#7f7f7f",
    )

    ax2.plot(
        esc_A["PeriodoYM"],
        esc_A["Precio_Pronosticado"],
        marker="o",
        linestyle="-",
        label=f"Escenario A (USD {usd_A:+}%, Chicago {fut_A:+}%)",
        color="#2ca02c",
    )

    ax2.plot(
        esc_B["PeriodoYM"],
        esc_B["Precio_Pronosticado"],
        marker="s",
        linestyle="-.",
        label=f"Escenario B (USD {usd_B:+}%, Chicago {fut_B:+}%)",
        color="#ff7f0e",
    )

    ax2.legend()
    ax2.set_ylabel("Precio (ARS constantes)")
    ax2.tick_params(axis="x", rotation=45)
    st.pyplot(fig2)

    st.subheader("⬇️ Descargar escenarios")

    esc_opcion = st.selectbox(
        "¿Qué escenario querés descargar?",
        ["Base", "Escenario A", "Escenario B"],
    )

    if esc_opcion == "Base":
        df_desc = forecast_base.copy()
        name = "pronostico_base_trigo.csv"
    elif esc_opcion == "Escenario A":
        df_desc = esc_A.copy()
        name = "pronostico_escenario_A_trigo.csv"
    else:
        df_desc = esc_B.copy()
        name = "pronostico_escenario_B_trigo.csv"

    csv_esc = df_desc.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Descargar {esc_opcion} (CSV)",
        data=csv_esc,
        file_name=name,
        mime="text/csv",
    )


# ---------------- TAB 3: MODELO Y MÉTRICAS ----------------
with tab3:
    st.subheader("Comparación de modelos (últimos 12 meses)")

    mae_naive = metrics["MAE_naive"]
    mae_arima = metrics["MAE_arima"]
    mae_hybrid = metrics["MAE_hybrid"]

    comp_df = pd.DataFrame(
        {
            "Modelo": ["Naïve (último valor)", "ARIMA solo", "Híbrido ARIMA + XGBoost"],
            "MAE (ARS/tn)": [mae_naive, mae_arima, mae_hybrid],
        }
    )

    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.markdown(
        """
- **Naïve:** asume que el próximo precio será igual al último observado.
- **ARIMA:** aprende solo a partir del historial de la serie.
- **Híbrido:** mejora el ARIMA corrigiendo sus errores con dólar, Chicago y estacionalidad.
"""
    )

    st.subheader("Importancia de variables en XGBoost")

    importances = xgb_model.feature_importances_
    imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    imp_series.tail(12).plot(kind="barh", ax=ax3, color="#2ca02c")
    ax3.set_xlabel("Importancia relativa")
    st.pyplot(fig3)

    with st.expander("Variables utilizadas (explicado simple)"):
        st.markdown(
            """
**Variables del mercado y del modelo:**

- **Precio del trigo en Argentina** (serie principal, en pesos constantes).
- **Futuro del trigo en Chicago** (referencia internacional de precios).
- **Tipo de cambio USD/ARS**.
- **Mes del año** (estacionalidad).
- **Tendencia temporal (t)**.
- **Residuos del ARIMA y sus lags (resid_lag1 ... resid_lag6)**.
"""
        )
