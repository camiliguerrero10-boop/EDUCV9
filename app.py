"""
SpVAR Colombia – Streamlit Cloud (sin geopandas/GDAL)
Mapa Folium robusto + Informe PDF con gráficos e interpretaciones
"""

import warnings
warnings.filterwarnings("ignore")

import json
import urllib.request
import io
import tempfile
import os
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from statsmodels.regression.linear_model import OLS as sm_OLS
from statsmodels.tools.tools import add_constant
from fpdf import FPDF

st.set_page_config(
    page_title="SpVAR Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
h1, h2, h3, p, label, .stMarkdown { color: #e2e8f0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


def lag_spatial_knn(values, coords, k=4):
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    out = np.zeros(len(values))
    for i in range(len(values)):
        out[i] = values[idx[i]].mean()
    return out


@st.cache_data(show_spinner="Cargando datos espaciales...")
def load_data():
    url = (
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/"
        "co_2018_MGN_DPTO_POLITICO.geojson"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        geojson = json.loads(r.read().decode())

    deps, lons, lats = [], [], []
    for feat in geojson["features"]:
        props = feat["properties"]
        name = props.get("DPTO_CNMBR") or props.get("departamento") or "N/A"
        deps.append(str(name).strip())
        geom = feat["geometry"]
        coords_list = []
        if geom["type"] == "Polygon":
            coords_list = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                coords_list.extend(poly[0])
        xs = [c[0] for c in coords_list]
        ys = [c[1] for c in coords_list]
        lons.append(float(np.mean(xs)))
        lats.append(float(np.mean(ys)))

    N = len(deps)
    coords = np.column_stack([lons, lats])

    np.random.seed(42)
    T = 10
    years = list(range(2015, 2015 + T))
    educ = np.zeros((T, N))
    growth = np.zeros((T, N))
    viirs = np.zeros((T, N))
    educ[0] = 8.2 + 1.8 * np.random.randn(N)
    growth[0] = 2.8 + 1.4 * np.random.randn(N)
    viirs[0] = 12 + 4.5 * np.random.randn(N)

    for t in range(1, T):
        educ[t] = (
            0.62 * educ[t - 1]
            + 0.19 * lag_spatial_knn(educ[t - 1], coords, k=4)
            + 0.09 * growth[t - 1]
            + 0.35 * np.random.randn(N)
        )
        growth[t] = (
            0.51 * growth[t - 1]
            + 0.21 * lag_spatial_knn(growth[t - 1], coords, k=4)
            + 0.28 * educ[t - 1]
            + 0.14 * lag_spatial_knn(educ[t - 1], coords, k=4)
            + 0.85 * np.random.randn(N)
        )
        viirs[t] = (
            0.70 * viirs[t - 1]
            + 0.15 * lag_spatial_knn(viirs[t - 1], coords, k=4)
            + 0.40 * growth[t]
            + 1.1 * np.random.randn(N)
        )

    df_long = pd.DataFrame(
        {
            "year": np.repeat(years, N),
            "departamento": np.tile(deps, T),
            "educacion": educ.flatten(),
            "crecimiento": growth.flatten(),
            "viirs": viirs.flatten(),
            "lon": np.tile(lons, T),
            "lat": np.tile(lats, T),
        }
    )
    df_long["crecimiento_pos"] = df_long["crecimiento"] - df_long["crecimiento"].min() + 1.5

    avg = (
        df_long.groupby("departamento")
        .agg(
            {
                "educacion": "mean",
                "crecimiento": "mean",
                "viirs": "mean",
                "lon": "first",
                "lat": "first",
            }
        )
        .reset_index()
    )

    for feat in geojson["features"]:
        name = feat["properties"].get("DPTO_CNMBR") or feat["properties"].get("departamento")
        name = str(name).strip()
        feat["properties"]["departamento"] = name
        row = avg[avg["departamento"] == name]
        if not row.empty:
            feat["properties"]["educacion"] = float(row["educacion"].iloc[0])
            feat["properties"]["crecimiento"] = float(row["crecimiento"].iloc[0])
            feat["properties"]["viirs"] = float(row["viirs"].iloc[0])

    df = df_long.sort_values(["departamento", "year"]).copy()
    df["educ_l1"] = df.groupby("departamento")["educacion"].shift(1)
    df["growth_l1"] = df.groupby("departamento")["crecimiento"].shift(1)
    df = df.dropna(subset=["educ_l1", "growth_l1"]).reset_index(drop=True)

    rows = []
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr].set_index("departamento").reindex(deps)
        e1 = sub["educ_l1"].values.astype(float)
        g1 = sub["growth_l1"].values.astype(float)
        vv = sub["viirs"].values.astype(float)
        We = lag_spatial_knn(e1, coords, k=4)
        Wg = lag_spatial_knn(g1, coords, k=4)
        for i, dep in enumerate(deps):
            rows.append(
                {
                    "year": yr,
                    "departamento": dep,
                    "educacion": float(sub.loc[dep, "educacion"]),
                    "crecimiento": float(sub.loc[dep, "crecimiento"]),
                    "educ_l1": e1[i],
                    "growth_l1": g1[i],
                    "W_educ_l1": We[i],
                    "W_growth_l1": Wg[i],
                    "viirs": vv[i],
                }
            )
    spvar_df = pd.DataFrame(rows)

    def estimate(y, X, names):
        m = sm_OLS(y, add_constant(X)).fit(cov_type="HC1")
        return dict(zip(["const"] + names, m.params))

    coef_e = estimate(
        spvar_df["educacion"].values,
        spvar_df[["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"]].values,
        ["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"],
    )
    coef_g = estimate(
        spvar_df["crecimiento"].values,
        spvar_df[["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"]].values,
        ["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"],
    )

    def impacts(rho, beta):
        rho = float(np.clip(rho, -0.9, 0.9))
        mult = 1.0 / (1.0 - rho) if abs(1.0 - rho) > 1e-6 else 1.0
        total = mult * beta
        return {"direct": beta, "indirect": total - beta, "total": total}

    imp_e = impacts(coef_e.get("W_educ_l1", 0.0), coef_e["educ_l1"])
    imp_g = impacts(coef_g.get("W_growth_l1", 0.0), coef_g["growth_l1"])

    return geojson, avg, df_long, spvar_df, coef_e, coef_g, imp_e, imp_g


def build_folium_map(geojson, avg):
    """Mapa Folium multi-capa robusto."""
    m = folium.Map(
        location=[4.6, -73.0],
        zoom_start=5.5,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # Capas coropléticas una por una
    for col, name, cmap in [
        ("viirs", "Luces nocturnas", "YlOrRd"),
        ("educacion", "Educación", "YlGnBu"),
        ("crecimiento", "Crecimiento", "RdYlGn"),
    ]:
        folium.Choropleth(
            geo_data=geojson,
            name=name,
            data=avg,
            columns=["departamento", col],
            key_on="feature.properties.departamento",
            fill_color=cmap,
            fill_opacity=0.7,
            line_opacity=0.3,
            line_weight=0.5,
            legend_name=name,
            nan_fill_color="gray",
        ).add_to(m)

    # Tooltips
    folium.GeoJson(
        geojson,
        name="Etiquetas",
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "transparent",
            "weight": 0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["departamento", "educacion", "crecimiento", "viirs"],
            aliases=["Departamento:", "Educación:", "Crecimiento:", "VIIRS:"],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    return m


def interpretaciones(coef_e, coef_g, imp_e, imp_g, avg, n_obs, ymin, ymax):
    """Texto interpretativo profesional para el informe."""
    lines = []

    # Autodinámica
    if coef_e["educ_l1"] > 0.5:
        lines.append(
            "La educacion muestra fuerte inercia temporal: los departamentos con "
            "mayor escolaridad tienden a mantener ese nivel en el tiempo."
        )
    else:
        lines.append(
            "La inercia educativa es moderada; hay espacio para politicas que "
            "alteren la trayectoria de escolaridad departamental."
        )

    if coef_g["growth_l1"] > 0.4:
        lines.append(
            "El crecimiento economico tambien es persistente: el desempeno pasado "
            "predice en buena medida el crecimiento presente."
        )
    else:
        lines.append(
            "El crecimiento es relativamente volatil; los shocks se disipan "
            "con mayor rapidez entre periodos."
        )

    # Cruce local
    if coef_g["educ_l1"] > 0:
        lines.append(
            f"Un aumento local de educacion se asocia con mayor crecimiento "
            f"(coef. {coef_g['educ_l1']:.3f}): hay evidencia de retorno economico "
            f"a la escolaridad dentro del mismo departamento."
        )
    else:
        lines.append(
            "En esta especificacion el vinculo local educacion-crecimiento "
            "es debil o negativo; conviene validar con datos oficiales DANE."
        )

    if coef_e["growth_l1"] > 0:
        lines.append(
            "El crecimiento local tambien retroalimenta la educacion, "
            "sugiriendo un circulo virtuoso regional."
        )

    # Spillovers
    if coef_g["W_educ_l1"] > 0:
        lines.append(
            f"Existe spillover educativo positivo hacia el crecimiento de vecinos "
            f"(coef. espacial cruzado {coef_g['W_educ_l1']:.3f}): invertir en "
            f"educacion en un departamento puede beneficiar a los colindantes."
        )
    else:
        lines.append(
            "Los spillovers educativos hacia el crecimiento de vecinos son "
            "limitados en esta muestra sintetica."
        )

    if coef_e["W_educ_l1"] > 0 or coef_g["W_growth_l1"] > 0:
        lines.append(
            "La dependencia espacial directa indica que los departamentos "
            "no evolucionan de forma aislada: el entorno regional importa."
        )

    # Impactos
    lines.append(
        f"Impacto total de largo plazo de la educacion (aproximacion): "
        f"{imp_e['total']:.3f} (directo {imp_e['direct']:.3f}, "
        f"indirecto/espacial {imp_e['indirect']:.3f})."
    )
    lines.append(
        f"Impacto total de largo plazo del crecimiento: "
        f"{imp_g['total']:.3f} (directo {imp_g['direct']:.3f}, "
        f"indirecto {imp_g['indirect']:.3f})."
    )

    # Ranking
    top_e = avg.nlargest(3, "educacion")["departamento"].tolist()
    low_e = avg.nsmallest(3, "educacion")["departamento"].tolist()
    top_g = avg.nlargest(3, "crecimiento")["departamento"].tolist()
    lines.append(
        f"Departamentos con mayor educacion promedio: {', '.join(top_e)}."
    )
    lines.append(
        f"Departamentos con menor educacion promedio: {', '.join(low_e)}."
    )
    lines.append(
        f"Mayor crecimiento promedio: {', '.join(top_g)}."
    )
    lines.append(
        f"Muestra: {n_obs} observaciones, periodo {ymin}-{ymax}. "
        f"Datos sinteticos con dependencia espacial; reemplazar por series "
        f"DANE/BanRep para politica publica."
    )
    return lines


def crear_pdf(avg, df_long, coef_e, coef_g, imp_e, imp_g, n_obs, ymin, ymax):
    """Genera PDF con portada, coeficientes, interpretaciones y graficos."""
    tmpdir = tempfile.mkdtemp()
    img1 = os.path.join(tmpdir, "barras.png")
    img2 = os.path.join(tmpdir, "scatter.png")
    img3 = os.path.join(tmpdir, "serie.png")

    # Grafico 1: top 10 educacion
    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = avg.nlargest(10, "educacion").sort_values("educacion")
    ax.barh(top["departamento"], top["educacion"], color="#38bdf8")
    ax.set_xlabel("Educacion promedio")
    ax.set_title("Top 10 departamentos – Educacion")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(img1, dpi=120, bbox_inches="tight")
    plt.close()

    # Grafico 2: educacion vs crecimiento
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(avg["educacion"], avg["crecimiento"], c=avg["viirs"], cmap="inferno", s=80, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Educacion promedio")
    ax.set_ylabel("Crecimiento promedio")
    ax.set_title("Educacion vs Crecimiento (color = luces nocturnas)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(img2, dpi=120, bbox_inches="tight")
    plt.close()

    # Grafico 3: evolucion temporal nacional
    nat = df_long.groupby("year")[["educacion", "crecimiento"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(nat.index, nat["educacion"], marker="o", label="Educacion", color="#38bdf8")
    ax.plot(nat.index, nat["crecimiento"], marker="s", label="Crecimiento", color="#4ade80")
    ax.set_xlabel("Anio")
    ax.set_title("Evolucion promedio nacional 2015-2024")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(img3, dpi=120, bbox_inches="tight")
    plt.close()

    interps = interpretaciones(coef_e, coef_g, imp_e, imp_g, avg, n_obs, ymin, ymax)

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(30, 41, 59)
            self.cell(0, 8, "SpVAR Colombia - Informe Ejecutivo", align="R", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 116, 139)
            self.cell(0, 8, f"Pagina {self.page_no()}/{{nb}}", align="C")

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Portada
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(15, 23, 42)
    pdf.ln(30)
    pdf.multi_cell(0, 12, "INFORME EJECUTIVO", align="C")
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, "SpVAR: Educacion y Crecimiento Economico", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, "Departamentos de Colombia · 2015-2024", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        7,
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Observaciones: {n_obs}\n"
        f"Matriz espacial: KNN k=4\n"
        f"Fuente geometria: GitHub raw (caticoa3/colombia_mapa)\n"
        f"Datos: sinteticos con dependencia espacial incorporada",
        align="C",
    )

    # Seccion 1: mecanismos
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Mecanismos de transmision espacial", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    texto_mec = (
        f"1. Autodinamica\n"
        f"   Educacion  phi_ee = {coef_e['educ_l1']:.4f}\n"
        f"   Crecimiento phi_gg = {coef_g['growth_l1']:.4f}\n\n"
        f"2. Cruce local\n"
        f"   Educ <- Crec local = {coef_e['growth_l1']:.4f}\n"
        f"   Crec <- Educ local = {coef_g['educ_l1']:.4f}\n\n"
        f"3. Relacion espacial directa\n"
        f"   W*Educ -> Educ = {coef_e['W_educ_l1']:.4f}\n"
        f"   W*Crec -> Crec = {coef_g['W_growth_l1']:.4f}\n\n"
        f"4. Relacion espacial cruzada\n"
        f"   W*Crec -> Educ = {coef_e['W_growth_l1']:.4f}\n"
        f"   W*Educ -> Crec = {coef_g['W_educ_l1']:.4f}\n\n"
        f"Impactos de largo plazo (aproximacion)\n"
        f"   Educacion  Directo={imp_e['direct']:.4f}  Indirecto={imp_e['indirect']:.4f}  Total={imp_e['total']:.4f}\n"
        f"   Crecimiento Directo={imp_g['direct']:.4f}  Indirecto={imp_g['indirect']:.4f}  Total={imp_g['total']:.4f}"
    )
    pdf.multi_cell(0, 5.5, texto_mec)

    # Seccion 2: interpretaciones
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Interpretacion de resultados", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for i, t in enumerate(interps, 1):
        # FPDF latin-1: evitar caracteres no soportados
        t_clean = (
            t.replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
            .replace("Á", "A").replace("É", "E").replace("Í", "I")
            .replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
        )
        pdf.multi_cell(0, 5.5, f"{i}. {t_clean}")
        pdf.ln(1)

    # Seccion 3: graficos
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Graficos de apoyo", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Figura 1. Top 10 departamentos por educacion promedio", new_x="LMARGIN", new_y="NEXT")
    pdf.image(img1, w=180)
    pdf.ln(4)
    pdf.cell(0, 6, "Figura 2. Relacion educacion-crecimiento (color = luces)", new_x="LMARGIN", new_y="NEXT")
    pdf.image(img2, w=160)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Graficos de apoyo (cont.)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Figura 3. Evolucion temporal promedio nacional", new_x="LMARGIN", new_y="NEXT")
    pdf.image(img3, w=180)
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Recomendaciones", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    recs = (
        "1. Priorizar inversion educativa en departamentos de baja escolaridad "
        "para reducir brechas regionales.\n"
        "2. Disenar politicas coordinadas entre vecinos, dado que existen "
        "spillovers espaciales.\n"
        "3. Validar estos resultados con series oficiales del DANE y BanRep "
        "antes de decisiones de politica publica.\n"
        "4. Complementar el SpVAR con variables de infraestructura, seguridad "
        "y capital humano avanzado."
    )
    pdf.multi_cell(0, 5.5, recs)
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(
        0,
        5,
        "Nota: este informe usa datos sinteticos realistas con dependencia espacial. "
        "No constituye evidencia causal definitiva para politica publica.",
    )

    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out.getvalue()


# ---- Carga ----
geojson, avg, df_long, spvar_df, coef_educ, coef_growth, imp_e, imp_g = load_data()

st.title("🇨🇴 SpVAR Colombia")
st.subheader("Educación y Crecimiento Económico · 33 departamentos · 2015-2024")
st.caption("Fuentes 100% públicas · Sin API keys · Compatible Streamlit Cloud")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Exploración Espacial", "🌌 Visualización 3D", "📊 SpVAR & Mecanismos", "📄 Informe PDF"]
)

with tab1:
    st.markdown("### Mapas coropléticos (polígonos)")
    if st.button("Generar mapas Plotly", type="primary", key="m1"):
        c1, c2, c3 = st.columns(3)

        def make_choro(col, title, scale):
            fig = px.choropleth(
                avg,
                geojson=geojson,
                locations="departamento",
                featureidkey="properties.departamento",
                color=col,
                color_continuous_scale=scale,
                title=title,
                hover_name="departamento",
            )
            fig.update_geos(fitbounds="locations", visible=False, bgcolor="#0f172a")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                margin=dict(l=0, r=0, t=40, b=0),
                height=420,
                font=dict(color="white"),
            )
            return fig

        with c1:
            st.plotly_chart(make_choro("educacion", "Educación", "YlGnBu"), use_container_width=True)
        with c2:
            st.plotly_chart(make_choro("crecimiento", "Crecimiento", "RdYlGn"), use_container_width=True)
        with c3:
            st.plotly_chart(make_choro("viirs", "Luces nocturnas", "Inferno"), use_container_width=True)

    st.markdown("### Mapa interactivo Folium (multi-capa)")
    st.caption("Use el control de capas (esquina superior derecha) para alternar Educación / Crecimiento / Luces.")
    if st.button("Cargar mapa Folium", type="primary", key="m2"):
        try:
            m = build_folium_map(geojson, avg)
            # returned_objects=[] evita re-renders; height fijo para Cloud
            st_folium(m, width=None, height=560, returned_objects=[], use_container_width=True)
        except Exception as e:
            st.error(f"Error al renderizar Folium: {e}")
            st.info("Alternativa: use los mapas Plotly de arriba o el informe PDF.")

with tab2:
    st.markdown("### Visualizaciones 3D")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Scatter 3D 2024", type="primary", key="b3d"):
            fig = px.scatter_3d(
                df_long[df_long["year"] == 2024],
                x="lon", y="lat", z="viirs",
                color="educacion", size="crecimiento_pos",
                hover_name="departamento",
                color_continuous_scale="Turbo", size_max=22,
                title="Educación × Crecimiento × Luces (2024)",
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", height=520,
                scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)),
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if st.button("Superficie luces", key="bsurf"):
            d = df_long[df_long["year"] == 2024]
            lon, lat, z = d["lon"].values, d["lat"].values, d["viirs"].values
            xi = np.linspace(lon.min(), lon.max(), 60)
            yi = np.linspace(lat.min(), lat.max(), 60)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((lon, lat), z, (Xi, Yi), method="cubic")
            fig = go.Figure(data=[go.Surface(x=Xi, y=Yi, z=Zi, colorscale="Inferno")])
            fig.update_layout(
                title="Superficie luces nocturnas", template="plotly_dark",
                paper_bgcolor="#0f172a", height=520,
                scene=dict(
                    xaxis_title="Lon", yaxis_title="Lat", zaxis_title="VIIRS",
                    aspectmode="manual", aspectratio=dict(x=1.3, y=1, z=0.45),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        if st.button("Animación 2015-2024", key="banim"):
            fig = px.scatter_3d(
                df_long, x="lon", y="lat", z="viirs",
                color="educacion", size="crecimiento_pos",
                animation_frame="year", hover_name="departamento",
                color_continuous_scale="Turbo", size_max=18,
                title="Evolución 2015-2024",
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", height=520,
                scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)),
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Cuatro mecanismos de transmisión espacial")
    if st.button("Mostrar mecanismos e interpretación", type="primary", key="bmec"):
        st.markdown(
            f"""
**1. Autodinámica espacial**  
• Educación φ_ee = **{coef_educ['educ_l1']:.4f}**  
• Crecimiento φ_gg = **{coef_growth['growth_l1']:.4f}**

**2. Cruce local**  
• Educ ← Crec local: **{coef_educ['growth_l1']:.4f}**  
• Crec ← Educ local: **{coef_growth['educ_l1']:.4f}**

**3. Relación espacial directa** (KNN k=4)  
• W·Educ → Educ: **{coef_educ['W_educ_l1']:.4f}**  
• W·Crec → Crec: **{coef_growth['W_growth_l1']:.4f}**

**4. Relación espacial cruzada**  
• W·Crec → Educ: **{coef_educ['W_growth_l1']:.4f}**  
• W·Educ → Crec: **{coef_growth['W_educ_l1']:.4f}**

---
### Impactos de largo plazo

| Variable | Directo | Indirecto | Total |
|----------|---------|-----------|-------|
| Educación | {imp_e['direct']:.4f} | {imp_e['indirect']:.4f} | {imp_e['total']:.4f} |
| Crecimiento | {imp_g['direct']:.4f} | {imp_g['indirect']:.4f} | {imp_g['total']:.4f} |
"""
        )
        st.markdown("### Interpretación")
        for t in interpretaciones(
            coef_educ, coef_growth, imp_e, imp_g, avg,
            len(spvar_df), spvar_df["year"].min(), spvar_df["year"].max(),
        ):
            st.write(f"• {t}")

with tab4:
    st.markdown("### Informe ejecutivo en PDF")
    st.write(
        "Incluye: portada, los 4 mecanismos, **interpretaciones**, "
        "**3 gráficos** (ranking, dispersión, evolución temporal) y recomendaciones."
    )
    if st.button("Generar informe PDF", type="primary", key="bpdf"):
        with st.spinner("Generando PDF..."):
            pdf_bytes = crear_pdf(
                avg, df_long, coef_educ, coef_growth, imp_e, imp_g,
                len(spvar_df), int(spvar_df["year"].min()), int(spvar_df["year"].max()),
            )
        st.success("PDF listo.")
        st.download_button(
            "📥 Descargar informe PDF",
            data=pdf_bytes,
            file_name="informe_spvar_colombia.pdf",
            mime="application/pdf",
            type="primary",
        )

st.markdown(
    '<div style="text-align:center;margin-top:2rem;color:#94a3b8;font-size:0.85rem;">'
    "SpVAR Colombia · Streamlit Cloud · Fuentes 100% públicas</div>",
    unsafe_allow_html=True,
)
