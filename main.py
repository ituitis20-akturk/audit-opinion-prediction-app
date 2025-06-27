
"""Streamlit app: upload an Excel file, compute **32 specific financial ratios**, then run an
XGB model that classifies “Görüş Tipi”.
──────────────────────────────────────────────────────────────────────────────
• User uploads **financial_data.xlsx** (one sheet, one row per firm/period).
• Column names follow Turkish IFRS naming (e.g. "Dönen Varlıklar", "Kısa Vadeli Yükümlülükler" …).
• Trained bundle **slim_xgb.joblib** (pipeline + LabelEncoder) must live in the
  same folder.

"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

###############################################################################
# 0. CONFIG & CONSTANTS
###############################################################################

st.set_page_config(page_title="Görüş Tipi Tahmin (32 Özellik)", layout="wide")
MODEL_PATH = "slim_xgb.joblib"  # → (pipeline, encoder)




# 32 finansal oran / metrik – model tam olarak bunları bekliyor
SELECTED_FEATS = [
    'Altman Z-Skoru', 'Finansal Kaldıraç', 'Nakit Oranı', 'Aktif Devir Hızı',
    'L Model Skoru', 'Zmijewski Skoru', 'Asit Test Oranı',
    'Özsermaye / Maddi Duran Varlıklar', 'Faaliyet Kar Marjı',
    'Duran Varlıklar / Maddi Özkaynak', 'Ticari Borçlar Devir Hızı',
    'Stok Devir Hızı', 'Brüt Kar Marjı (%)', 'Cari Oran',
    'Esas Faaliyet Karı / Kısa Vadeli Borç', 'Esas Faaliyet Kar Marjı',
    'Özsermaye / Aktif', 'Alacak Devir Hızı', 'Borç Kaynak Oranı',
    'Net Kar Marjı', 'Net Satışlar / Kısa Vade Borç',
    'Kısa Vade Borç / Özsermaye', 'Kısa Vade Borç / Toplam Borç',
    'Kısa Vade Borç / Aktif', 'Kısa Vade Borç / Dönen Varlık',
    'FAVÖK / Kısa Vade Borç', 'Duran Varlıklar / Aktif ',
    'Dönen Varlıklar / Aktif (%)', 'Dönen Varlıklar Devir Hızı',
    'Aktif Karlılık (%)', 'ROCE Oranı', 'Finansman Gider / Net Satış'
]

###############################################################################
# 1. HELPER FUNCTIONS
###############################################################################

def safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """
     Payda 0 (veya NaN) ise sonuç 0 döner.
     Aksi hâlde normal bölüm değeri döner.
    """
    denom_replaced = denom.replace(0, np.nan)
    result = num / denom_replaced
    return result.fillna(0)



def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all 32 ratios & add them as new columns to *df*."""
    # --- 1) Standardise header names -------------------------------------------------


    # --- 2) Common aggregates (avoid recalculating) ----------------------------------
    total_assets = df['Dönen Varlıklar'] + df['Duran Varlıklar']
    total_liab   = df['Kısa Vadeli Yükümlülükler'] + df['Uzun Vadeli Yükümlülükler']

    # --- 3) Liquidity ratios ----------------------------------------------------------
    df['Cari Oran']              = safe_div(df['Dönen Varlıklar'], df['Kısa Vadeli Yükümlülükler'])
    df['Asit Test Oranı']        = safe_div(df['Dönen Varlıklar'] - df['Stoklar'] - df['Diğer Dönen Varlıklar'],
                                            df['Kısa Vadeli Yükümlülükler'])
    df['Nakit Oranı']            = safe_div(df['Nakit ve Nakit Benzerleri'], df['Kısa Vadeli Yükümlülükler'])

    # --- 4) Profitability & margins ---------------------------------------------------
    df['Faaliyet Kar Marjı']     = safe_div(df['FAALİYET KARI (ZARARI)']*100, df['Satış Gelirleri'])
    df['Esas Faaliyet Kar Marjı']= safe_div(df['Net Faaliyet Kar/Zararı']*100, df['Satış Gelirleri'])
    df['Brüt Kar Marjı (%)']     = safe_div(df['Ticari Faaliyetlerden Brüt Kar (Zarar)']*100, df['Satış Gelirleri'])
    df['Net Kar Marjı']          = safe_div(df['Dönem Net Kar/Zararı']*100, df['Satış Gelirleri'])
    df['FAVÖK / Kısa Vade Borç'] = safe_div(df['FAALİYET KARI (ZARARI)'], df['Kısa Vadeli Yükümlülükler'])

    # --- 5) Turnover ratios -----------------------------------------------------------
    df['Aktif Devir Hızı']           = safe_div(df['Satış Gelirleri'], total_assets)
    df['Alacak Devir Hızı']          = safe_div(df['Satış Gelirleri'], df['Ticari Alacaklar'])
    df['Dönen Varlıklar Devir Hızı'] = safe_div(df['Dönen Varlıklar'], df['Satış Gelirleri'])
    df['Ticari Borçlar Devir Hızı']  = -safe_div(df['Satışların Maliyeti (-)'], df['Ticari Borçlar'])
    df['Stok Devir Hızı']            = -safe_div(df['Satışların Maliyeti (-)'], df['Stoklar'])

    # --- 6) Capital structure ratios --------------------------------------------------
    df['Borç Kaynak Oranı']                 = safe_div(total_liab, df['Toplam Özkaynaklar'])*100
    df['Finansal Kaldıraç']                 = safe_div(total_liab, total_assets)*100
    df['Özsermaye / Aktif']                = safe_div(df['Toplam Özkaynaklar'], total_assets)
    df['Özsermaye / Maddi Duran Varlıklar'] = safe_div(df['Toplam Özkaynaklar'], df['Maddi Duran Varlıklar'])
    df['Duran Varlıklar / Maddi Özkaynak']  = safe_div(df['Duran Varlıklar'],
                                                       df['Toplam Özkaynaklar'] - df['Maddi Olmayan Duran Varlıklar'])
    df['Duran Varlıklar / Aktif ']          = safe_div(df['Duran Varlıklar']*100, total_assets)
    df['Dönen Varlıklar / Aktif (%)']       = safe_div(df['Dönen Varlıklar']*100, total_assets)

    # --- 7) Short‑term debt focus -----------------------------------------------------
    df['Kısa Vade Borç / Aktif']          = safe_div(df['Kısa Vadeli Yükümlülükler'], total_assets)
    df['Kısa Vade Borç / Dönen Varlık']   = safe_div(df['Kısa Vadeli Yükümlülükler'], df['Dönen Varlıklar'])
    df['Kısa Vade Borç / Özsermaye']      = safe_div(df['Kısa Vadeli Yükümlülükler'], df['Toplam Özkaynaklar'])
    df['Kısa Vade Borç / Toplam Borç']    = safe_div(df['Kısa Vadeli Yükümlülükler'], total_liab)
    df['Net Satışlar / Kısa Vade Borç']   = safe_div(df['Satış Gelirleri'], df['Kısa Vadeli Yükümlülükler'])
    df['Esas Faaliyet Karı / Kısa Vadeli Borç'] = safe_div(df['Net Faaliyet Kar/Zararı'], df['Kısa Vadeli Yükümlülükler'])

    # --- 8) Misc. profitability -------------------------------------------------------
    df['Aktif Karlılık (%)']  = safe_div(df['Dönem Net Kar/Zararı']*100, total_assets)
    df['ROCE Oranı']          = safe_div(df['FAALİYET KARI (ZARARI)']*100, total_assets)
    df['Finansman Gider / Net Satış'] = safe_div(df['Finansman Giderleri'], df['Satış Gelirleri'])

    # --- 9) Bankruptcy / scoring models ----------------------------------------------
    # Altman Z inputs
    X1 = safe_div(df['Dönen Varlıklar'] - df['Kısa Vadeli Yükümlülükler'], total_assets)
    X2 = safe_div(df['Geçmiş Yıllar Kar/Zararları'] + df['Dönem Net Kar/Zararı'], total_assets)
    X3 = safe_div(df['SÜRDÜRÜLEN FAALİYETLER VERGİ ÖNCESİ KARI (ZARARI)'], total_assets)
    X4 = safe_div(df['Toplam Özkaynaklar'], total_liab)
    X5 = safe_div(df['Satış Gelirleri'], total_assets)
    df['Altman Z-Skoru'] = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + X5

    # Zmijewski
    Z1 = safe_div(df['Dönem Net Kar/Zararı'], total_assets)
    Z2 = safe_div(total_liab, total_assets)
    Z3 = safe_div(df['Dönen Varlıklar'], df['Kısa Vadeli Yükümlülükler'])
    df['Zmijewski Skoru'] = -4.3 - 4.5*Z1 + 5.7*Z2 - 0.004*Z3

    # L‑model (uses X1–X5 again plus extra liquidity ratios)
    L6 = safe_div(safe_div(df['Nakit ve Nakit Benzerleri'], df['Kısa Vadeli Yükümlülükler']), total_liab)
    L7 = safe_div(total_liab, total_assets)
    df['L Model Skoru'] = (
        -0.113*X1 + 0.238*X2 - 0.052*X3 - 0.051*X4 + 0.011*X5 + 0.729*L6 - 0.639*L7
    )

    return df




###############################################################################
# 2 + 3.  Excel’i oku  →  tablola (başlık-transpoz)  →  oranları hesapla
#         →  modele ver  →  tahmin + indirme
###############################################################################

st.sidebar.header("1️⃣ Excel Yükle")
file = st.sidebar.file_uploader(
    "Finansal tablo (.xlsx /.xls)", type=["xlsx", "xls", "xlsm"]
)

if not file:
    st.info("⬅️ Lütfen analiz edilecek Excel dosyasını seçin.")
    st.stop()

# --------------------------------------------------------------------------- #
# 2️⃣  VERTICAL → HORIZONTAL dönüştürme  (senin eski kodun bire bir)
# --------------------------------------------------------------------------- #
raw_vert = pd.read_excel(BytesIO(file.read()), header=None, sheet_name="Sheet1")
raw_df = (
    raw_vert
    .set_index(0)          # Sol 1. sütun başlık oluyor
    .T                     # Transpoz: başlıklar sütun adı
    .rename_axis(None)     # İsim yok
    .reset_index(drop=True)
)
raw_df.columns = raw_df.columns.str.strip()           # Gereksiz boşluk ayıkları
raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]  # Çift başlık varsa at

st.write("### Dönüştürülmüş Veri", raw_df.head())

# --------------------------------------------------------------------------- #
# 3️⃣  Oranları hesapla ve 32 özelliği çıkar
# --------------------------------------------------------------------------- #
with st.spinner("Oranlar hesaplanıyor..."):
    enriched = compute_ratios(raw_df)             # 32 oran eklenir
    model_input = enriched[SELECTED_FEATS].copy()

# Eksik satırları temizle
if model_input.isna().any().any():
    st.warning("Bazı oranlar eksik → ilgili satırlar atlandı.")
    model_input.dropna(inplace=True)
if model_input.empty:
    st.error("Hiç analiz edilebilir satır kalmadı.")
    st.stop()

# --------------------------------------------------------------------------- #
#  Modeli yükle  →  tahmin et  →  sonuçları göster + indir
# --------------------------------------------------------------------------- #
try:
    pipe, enc = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
    st.stop()

pred_num   = pipe.predict(model_input)
pred_label = enc.inverse_transform(pred_num)

out_df = raw_df.loc[model_input.index].copy()
out_df["Tahmin Görüş Tipi"] = pred_label

st.success(f"{len(out_df)} satır başarıyla tahmin edildi.")
st.write("### Tahmin Sonuçları", out_df[["Tahmin Görüş Tipi"]].head())

@st.cache_data
def to_xlsx(df: pd.DataFrame) -> bytes:
    with BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name="sonuclar")
        return buf.getvalue()

st.download_button(
    "📥 Sonuçları Excel olarak indir",
    data=to_xlsx(out_df),
    file_name="tahmin_sonuclari.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("🔍 Kullanılan 32 Özellik"):
    st.write(model_input.head())
