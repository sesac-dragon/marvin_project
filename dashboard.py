import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sqlalchemy import create_engine
import pymysql


st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] * {
        font-size: 20px !important;
        font-family: "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif !important;
    }
    th, td, .css-10trblm, .css-1cpxqw2 {
        font-size: 15px !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-size: 1.5em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ==========================================================

with st.sidebar:
    page = st.selectbox('페이지', ['엔카 매물 추천', '엔카 신뢰도 분석'])

if page == '엔카 매물 추천':

    st.markdown(
        "<h1 style='font-size: 35px; color: #1A3C8B; font-weight:bold;'>엔카 매물 추천</h1>",
        unsafe_allow_html=True
    )

    def load_config_from_txt(filename):
        config = {}
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    config[k.strip()] = v.strip()
        return config

    DB_CONFIG = load_config_from_txt("marvin_project_db.config.txt")

    def get_db_engine():
        return create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

    indicators_fields = [
        'ownerChangeCnt', 'Price', 'perf_issue_cnt', 'Mileage', 'FormYear',
        'total_accidents', 'total_repair_cost', 'notjoin_any',
        'type1_cnt', 'type2_cnt', 'type3_cnt', 'no_history_or_inspect_ratio'
    ]
    indicators_names = [
        "소유자 변경 횟수", "가격", "성능점검 특이사항 건수", "주행거리", "연식",
        "총 사고건수", "총 보험수리비", "자차 보험 미가입 비율(%)",
        "경미 사고 건수(Type1)", "보통 사고 건수(Type 2)", "중대 사고 건수(Type 3)",
        "보험/성능 미공개 비율(%)"
    ]
    weights = [1.2, 1.0, 1.4, 1.6, 1.7, 1.7, 2.0, 2.5, 1.0, 1.7, 3.0, 2.5]
    reverse_fields = [
        True, True, True, True, False, True, True, True, True, True, True, True
    ]

    @st.cache_data(show_spinner=False)
    def load_all_data():
        engine = get_db_engine()
        nomal = pd.read_sql(
            "SELECT vehicleNo, vehicleId, Manufacturer, ModelGroup, Model, FormYear, Mileage, Price, Color FROM nomal_info_data", engine)
        hist = pd.read_sql(
            "SELECT vehicleNo, ownerChangeCnt, myAccidentCnt, otherAccidentCnt, myAccidentCost, otherAccidentCost, notJoinDate1, notJoinDate2, notJoinDate3, notJoinDate4, notJoinDate5 FROM insur_history_data", engine)
        insp_cols_df = pd.read_sql("SHOW FULL COLUMNS FROM inspection_data", engine)
        perf_field_patterns = [
            '자가진단', '원동기', '변속기', '동력전달', '조향', '제동', '전기', '연료', '외관', '수리필요', '기본품목'
        ]
        perf_field_list = ['vehicleId', 'comments'] + [
            col for col in insp_cols_df['Field']
            if any(pat in col for pat in perf_field_patterns)
        ]
        cols_with_backticks = [f'`{col}`' for col in perf_field_list]
        sql = f"SELECT {', '.join(cols_with_backticks)} FROM inspection_data"
        insp = pd.read_sql(sql, engine)
        isur_detail = pd.read_sql("SELECT vehicleNo, type FROM insur_detail_data", engine)
        insur_nomal = pd.read_sql("SELECT vehicleNo FROM insur_nomal_data", engine)
        merged = nomal.merge(hist, on='vehicleNo', how='left') \
            .merge(insp, on='vehicleId', how='left')
        return merged, isur_detail, insur_nomal, insp

    df, isur_detail, insur_nomal, insp = load_all_data()

    for col in ['ownerChangeCnt', 'Price', 'Mileage', 'FormYear', 'myAccidentCnt', 'otherAccidentCnt', 'myAccidentCost', 'otherAccidentCost']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    perf_cols = [col for col in df.columns if any(x in col for x in
                                                  ['자가진단', '원동기', '변속기', '동력전달', '조향', '제동', '전기', '연료', '외관', '수리필요', '기본품목'])]
    def is_issue(val):
        return pd.notnull(val) and str(val).strip() not in ['양호', '적정', '', 'None', '없음', 'nan']
    df['perf_issue_cnt'] = df[perf_cols].apply(lambda row: sum(is_issue(v) for v in row), axis=1)
    df['total_accidents'] = df['myAccidentCnt'].fillna(0) + df['otherAccidentCnt'].fillna(0)
    df['total_repair_cost'] = df['myAccidentCost'].fillna(0) + df['otherAccidentCost'].fillna(0)

    notjoin_cols = [c for c in ['notJoinDate1', 'notJoinDate2', 'notJoinDate3', 'notJoinDate4', 'notJoinDate5'] if c in df.columns]
    df['notjoin_any'] = (df[notjoin_cols].notnull().any(axis=1).astype(float) * 100) if notjoin_cols else 0

    for t, cname in zip(['1', '2', '3'], ['type1_cnt', 'type2_cnt', 'type3_cnt']):
        t_count = isur_detail[isur_detail['type'] == t].groupby('vehicleNo').size()
        df[cname] = df['vehicleNo'].map(t_count).fillna(0).astype(int)

    def calc_nohist_or_inspect_ratio(subdf, insur_nomal, insp):
        insur_vno_set = set(insur_nomal['vehicleNo'].astype(str).dropna())
        insp_vid_set = set(insp['vehicleId'].astype(str).dropna())
        cnt_all = len(subdf)
        cnt_missing = 0
        for _, row in subdf.iterrows():
            vno = str(row['vehicleNo']) if 'vehicleNo' in row else None
            vid = str(row['vehicleId']) if 'vehicleId' in row else None
            if (vno not in insur_vno_set) or (vid not in insp_vid_set):
                cnt_missing += 1
        return (cnt_missing / cnt_all * 100) if cnt_all > 0 else 0.0


    def car_category_selector(label_prefix):
        manu = st.selectbox(f"{label_prefix} 제조사", sorted(df["Manufacturer"].dropna().unique()), key=f"{label_prefix}_manu")
        group_list = sorted(df[df["Manufacturer"] == manu]["ModelGroup"].dropna().unique())
        group = st.selectbox(f"{label_prefix} 모델그룹", group_list, key=f"{label_prefix}_group")
        model_list = sorted(df[(df["Manufacturer"] == manu) & (df["ModelGroup"] == group)]["Model"].dropna().unique().tolist())
        model_options = ["전체"] + model_list
        model = st.selectbox(f"{label_prefix} 모델", model_options, key=f"{label_prefix}_model")
        return {'Manufacturer': manu, 'ModelGroup': group, 'Model': model}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1비교군")
        cat1 = car_category_selector("1번")
    with col2:
        st.subheader("2비교군")
        cat2 = car_category_selector("2번")

    def filter_cars(cat):
        f = (df['Manufacturer'] == cat['Manufacturer']) & (df['ModelGroup'] == cat['ModelGroup'])
        if cat['Model'] != "전체":
            f = f & (df['Model'] == cat['Model'])
        return df[f].copy()

    df1 = filter_cars(cat1)
    df2 = filter_cars(cat2)
    n1 = len(df1)
    n2 = len(df2)

    def zscore_good(val, dist, reverse=False):
        arr = pd.to_numeric(val, errors='coerce')
        distarr = pd.to_numeric(dist, errors='coerce')
        mean, std = np.nanmean(distarr), np.nanstd(distarr)
        zs = np.zeros(len(arr)) if std == 0 else (arr - mean) / std
        return -zs if reverse else zs

    def get_weighted_zscore_vector(sub_df, all_df, insur_nomal, insp, scale=1.0):  
        zvec = []
        for i, f in enumerate(indicators_fields[:-1]):
            v = pd.to_numeric(sub_df[f], errors='coerce').dropna()
            allv = pd.to_numeric(all_df[f], errors='coerce').dropna()
            zval = 0 if (len(v) == 0 or len(allv) == 0) else zscore_good([v.mean()], allv, reverse=reverse_fields[i])[0]
            zvec.append(zval * weights[i] * scale)
        ratio = calc_nohist_or_inspect_ratio(sub_df, insur_nomal, insp)
        all_ratios = []
        for manu in all_df['Manufacturer'].dropna().unique():
            manu_df = all_df[all_df['Manufacturer'] == manu]
            all_ratios.append(calc_nohist_or_inspect_ratio(manu_df, insur_nomal, insp))
        mean, std = np.mean(all_ratios), np.std(all_ratios) if len(all_ratios) else (0, 1)
        z_ratio = (ratio - mean) / std if std > 0 else 0
        if reverse_fields[-1]:
            z_ratio = -z_ratio
        zvec.append(z_ratio * weights[-1] * scale)
        return zvec, ratio


    zvec1, df1_ratio = get_weighted_zscore_vector(df1, df, insur_nomal, insp)
    zvec2, df2_ratio = get_weighted_zscore_vector(df2, df, insur_nomal, insp)

    C_COLORS = ['#3b75af', '#d8713a']

    labels_with_count = [
        f"{cat1['Manufacturer']} {cat1['ModelGroup']} {cat1['Model'] if cat1['Model'] != '전체' else ''} (1) [{n1}대 집계]",
        f"{cat2['Manufacturer']} {cat2['ModelGroup']} {cat2['Model'] if cat2['Model'] != '전체' else ''} (2) [{n2}대 집계]"
    ]
    labels = [
        f"{cat1['Manufacturer']} {cat1['ModelGroup']} {cat1['Model'] if cat1['Model'] != '전체' else ''} (1)",
        f"{cat2['Manufacturer']} {cat2['ModelGroup']} {cat2['Model'] if cat2['Model'] != '전체' else ''} (2)"
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=zvec1, theta=indicators_names, fill='toself', name=labels[0], line_color=C_COLORS[0]
    ))
    fig.add_trace(go.Scatterpolar(
        r=zvec2, theta=indicators_names, fill='toself', name=labels[1], line_color=C_COLORS[1]
    ))
    fig.update_layout(
        width=1100, 
        height=750,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-10, 2]
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.50,
            xanchor="left",
            x=0.02,
            font=dict(size=50),
            itemwidth=100,
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    score1 = sum(zvec1)
    score2 = sum(zvec2)
    st.write("### 종합 상태점수")
    st.markdown(
        f"""
        - **{labels_with_count[0]}**: {score1:,.2f}
        - **{labels_with_count[1]}**: {score2:,.2f}
        """
    )
    if score1 > score2:
        st.success(f"**{labels_with_count[0]}** 가 더 매물 상태가 우수합니다.")
    elif score2 > score1:
        st.success(f"**{labels_with_count[1]}** 가 더 매물 상태가 우수합니다.")
    else:
        st.info("두 집단의 매물 상태가 거의 유사합니다!")

    def format_value(val, decimal_places, comma):
        import pandas as pd
        if pd.isna(val):
            return ""
        try:
            if decimal_places == 0:
                val_int = int(round(val))
                return f"{val_int:,}" if comma else f"{val_int}"
            else:
                val_float = round(float(val), decimal_places)
                if comma:
                    integer_part = int(val_float)
                    decimal_part = abs(val_float - integer_part)
                    formatted_decimal = f"{decimal_part:.{decimal_places}f}"[1:]
                    formatted_integer = f"{integer_part:,}"
                    return formatted_integer + formatted_decimal
                else:
                    return f"{val_float:.{decimal_places}f}"
        except Exception as e:
            return str(val)

    def format_table_values(df, labels_with_count):
        format_map = {
            "총 사고건수": (1, True),
            "총 보험수리비": (0, True),
            "중대 사고 건수(Type 3)": (1, True),
            "주행거리": (0, True),
            "자차 보험 미가입 비율(%)": (1, True),
            "연식": (0, False),
            "소유자 변경 횟수": (1, True),
            "성능점검 특이사항 건수": (1, True),
            "보험/성능 미공개 비율(%)": (1, True),
            "보통 사고 건수(Type 2)": (1, True),
            "경미 사고 건수(Type1)": (1, True),
            "가격": (0, True)
        }
        colnames = [
            f"{labels_with_count[0]} 평균",
            "엔카 전체 매물 평균",
            f"{labels_with_count[1]} 평균"
        ]
        for item, (decimals, comma) in format_map.items():
            for col in colnames:
                if item in df.index and col in df.columns:
                    df.at[item, col] = format_value(df.at[item, col], decimals, comma)
        return df

    enka_avg_list = [df[f].mean() for f in indicators_fields[:-1]] + [calc_nohist_or_inspect_ratio(df, insur_nomal, insp)]

    tab = pd.DataFrame({
        '항목': indicators_names,
        f"{labels_with_count[0]} 평균": (
            [df1[f].mean() if n1 > 0 else np.nan for f in indicators_fields[:-1]] + [df1_ratio]
        ),
        "엔카 전체 매물 평균": enka_avg_list,
        f"{labels_with_count[1]} 평균": (
            [df2[f].mean() if n2 > 0 else np.nan for f in indicators_fields[:-1]] + [df2_ratio]
        )
    }).set_index('항목')
    tab = format_table_values(tab, labels_with_count)
    st.write("#### 비교 항목별 원본 평균")
    st.dataframe(tab, use_container_width=True)

    st.info(
        """
        경미 사고 (Type 1): 차량 표면의 도장(페인트)만 벗겨짐 등 경미한 손상 \n
        보통 사고 (Type 2): 차량의 판금/교환/부품 교체가 일부 포함되는 손상 \n
        중대 사고 (Type 3): 차량의 주요 구조부(프레임/샤시) 손상 및 사고 \n
        """
    )

    def calc_state_score(row, ref_df, insur_nomal, insp, z_ratio):
        zvals = []
        for i, f in enumerate(indicators_fields[:-1]):
            allv = pd.to_numeric(ref_df[f], errors='coerce').dropna()
            val = row[f] if pd.notnull(row[f]) else np.nan
            zv = 0 if len(allv) == 0 or pd.isnull(val) else zscore_good([val], allv, reverse=reverse_fields[i])[0] * weights[i]
            zvals.append(zv)
        zvals.append(z_ratio)
        return np.sum(zvals)

    def filter_for_recommendation(df_grp, insur_nomal, insp):
        f_join = (df_grp['notjoin_any'] == 0) | (df_grp['notjoin_any'].isnull())
        insur_vnos = set(insur_nomal['vehicleNo'].astype(str))
        insp_vnos = set(insp['vehicleId'].astype(str))
        vnos = df_grp['vehicleNo'].astype(str)
        f_hist_or_insp = vnos.apply(lambda x: (x in insur_vnos) or (x in insp_vnos))
        return df_grp[f_join & f_hist_or_insp]

    def get_topn(df_grp, ref_df, insur_nomal, insp, z_ratio, N=5):
        scores = df_grp.apply(lambda r: calc_state_score(r, ref_df, insur_nomal, insp, z_ratio), axis=1)
        temp = df_grp.copy()
        temp['상태점수'] = scores
        topn = temp.sort_values('상태점수', ascending=False).head(N)
        cols = ['vehicleNo', 'FormYear', 'Mileage', 'ownerChangeCnt', 'Price', 'total_accidents', 'total_repair_cost', '상태점수']
        rename_map = {
            'vehicleNo': '차번호',
            'FormYear': '연식',
            'Mileage': '주행거리(km)',
            'ownerChangeCnt': '소유자변경횟수',
            'Price': '가격',
            'total_accidents': '총 보험수리 건수',
            'total_repair_cost': '총 보험수리 금액(원)',
            '상태점수': '상태점수'
        }
        disp = topn[cols].rename(columns=rename_map)
        disp = disp.reset_index(drop=True)
        disp.insert(0, '순위', range(1, len(disp) + 1))

        float0_cols_nocomma = ['연식']
        for col in float0_cols_nocomma:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")

        float0_cols_comma = ['주행거리(km)', '가격', '총 보험수리 건수', '총 보험수리 금액(원)']
        for col in float0_cols_comma:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "")

        float1_cols = ['소유자변경횟수']
        for col in float1_cols:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
            if '상태점수' in disp.columns:
                disp['상태점수'] = disp['상태점수'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        return disp

    st.markdown(f"#### {labels_with_count[0]} 추천 Top5")
    if n1 > 0:
        df1_filtered = filter_for_recommendation(df1, insur_nomal, insp)
        st.dataframe(
            get_topn(df1_filtered, df, insur_nomal, insp, zvec1[-1]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("첫 번째 비교군에 해당하는 매물이 없습니다.")

    st.markdown(f"#### {labels_with_count[1]} 추천 Top5")
    if n2 > 0:
        df2_filtered = filter_for_recommendation(df2, insur_nomal, insp)
        st.dataframe(
            get_topn(df2_filtered, df, insur_nomal, insp, zvec2[-1]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("두 번째 비교군에 해당하는 매물이 없습니다.")

    st.info(
        """
        - 각 항목의 z-score는 0값은 크롤링된 전체 엔카 매물 평균을 의미합니다.
        - 방사형 그래프가 넓은 비교군이 상태가 더 좋은 비교군입니다.
        - '상태점수'는 방사형 그래프 모든 항목을 가중치를 적용하여 평가한 종합 점수입니다.
        - Top5 추천매물는 방사형 그래프의 넓이, 즉 상태점수를 반영해 우수 매물을 선정합니다.
        """
    )

################################################################################################
elif page == '엔카 신뢰도 분석':

    def load_db_config(config_path='marvin_project_db.config.txt'):
        config = {}
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    config[k.strip()] = v.strip()
        return config

    def get_connection(cfg):
        return pymysql.connect(
            host=cfg['host'],
            user=cfg['user'],
            password=cfg['password'],
            db=cfg['database'],
            port=int(cfg['port']),
            charset='utf8'
        )

    INSUR_TYPE_DESC = {"1": "경미", "2": "보통", "3": "중대"}

    @st.cache_data(show_spinner=False)
    def load_all_reliability(_conn):
        df_nomal = pd.read_sql(
            "SELECT vehicleNo, vehicleId, Manufacturer, Model, Price, DealerPhoto FROM nomal_info_data", _conn)
        df_all = pd.read_sql("""
            SELECT n.vehicleNo, n.vehicleId, d.type as max_type, f.frame_damage_flag, i.accdient
            FROM nomal_info_data n
            LEFT JOIN (
                SELECT vehicleNo, MAX(type) as type FROM insur_detail_data GROUP BY vehicleNo
            ) d ON n.vehicleNo = d.vehicleNo
            LEFT JOIN inspection_frame_data f ON n.vehicleId = f.vehicleId
            LEFT JOIN inspection_data i ON n.vehicleId = i.vehicleId
        """, _conn)
        df = pd.merge(df_nomal, df_all, on=['vehicleNo', 'vehicleId'], how='left')

        def calc_reliability(row):
            if pd.isnull(row['max_type']):
                return 40, "D : 보험이력 미공개"
            if pd.isnull(row['frame_damage_flag']):
                return 40, "D : 성능점검 이력 미공개"
            max_type = str(row['max_type']) if not pd.isnull(row['max_type']) else None
            insp = int(row['accdient']) if not pd.isnull(row['accdient']) else 0
            frame = int(row['frame_damage_flag']) if not pd.isnull(row['frame_damage_flag']) else 0
            if max_type == '3' and frame == 0:
                return 10, "D : 보험상 중대사고이나 성능점검표에 프레임 손상 이력이 없음(은폐/누락 가능성)"
            elif max_type == '3' and frame == 1:
                return 80, "B : 보험상 중대사고이며 성능점검표에 프레임 손상 이력이 정상적으로 표기됨(공개)"
            elif max_type == '2' and insp == 0 and frame == 0:
                return 60, "C : 보험상 보통사고이나 점검표상 무사고로 표기됨"
            elif max_type == '1' and (insp == 1 or frame == 1):
                return 80, "B : 경미사고임에도 점검표에는 사고/프레임 손상 있음으로 표기(경미 불일치)"
            elif max_type == '2' and (insp == 1 or frame == 1):
                return 80, "B : 보통사고인데 점검표에 일부만 반영(부분 불일치)"
            elif max_type in ['1'] and insp == 0 and frame == 0:
                return 95, "A : 경미사고와 점검표 정보가 일치함"
            elif (insp == 1 or frame == 1) and (max_type in ['1', '2'] or max_type is None):
                return 95, "A : 점검표에 사고/프레임 손상 기록이 정상적으로 반영됨"
            elif max_type is None:
                return 100, "A : 사고 및 점검 이력이 모두 없는 클린카"
            else:
                return 80, "B : 부분 불일치 또는 기타(모호)"

        df[['신뢰도점수', '신뢰사유']] = df.apply(lambda row: pd.Series(calc_reliability(row)), axis=1)
        def get_grade(score, reason):
            if score >= 95:
                return 'A (안심)'
            elif score >= 80:
                return 'B (표준)'
            elif score >= 50:
                return 'C (불일치)'
            else:
                return 'D (매우 주의)'
        df['등급'] = df.apply(lambda row: get_grade(row['신뢰도점수'], row['신뢰사유']), axis=1)
        df['딜러식별'] = df['DealerPhoto']
        return df

    def main():
        st.title("엔카 신뢰도 분석")

        cfg = load_db_config()
        conn = get_connection(cfg)
        df = load_all_reliability(conn)

        st.subheader("전체 매물 신뢰도 등급 분포")
        grade_counts = df['등급'].value_counts().reset_index()
        grade_counts.columns = ['등급', '매물수']
        fig = px.pie(
            grade_counts,
            values='매물수',
            names='등급',
            hole=0.55,
            color='등급',
            color_discrete_map={
                'A (안심)': "#63b359",
                'B (표준)': "#4eb5eb",
                'C (불일치)': "#fdde47",
                'D (매우 주의)': "#ec5656"
            }
        )
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(showlegend=False) 
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <b style="color:#63b359">A (안심):</b> 보험·성능점검 모두 이력이 일치하거나, 사고이력 없는 매물.<br>
        <b style="color:#4eb5eb">B (표준):</b> 경미 사고(Type1) 성능점검표 누락.<br>
        <b style="color:#fdde47">C (불일치):</b> 보통사고(Type2) 성능점검표 누락.<br>
        <b style="color:#ec5656">D (매우 주의):</b> 보험 또는 점검 이력이 미공개이거나, 중대사고(Type3) 성능점검표 누락.
        """, unsafe_allow_html=True)


        dealer_counts = (
            df.groupby('딜러식별')
            .agg(
                총매물수=('vehicleNo', 'count'),
                D등급매물수=('등급', lambda x: (x == 'D (매우 주의)').sum())
            )
            .reset_index()
        )
        dealer_counts['D등급비율(%)'] = (dealer_counts['D등급매물수'] / dealer_counts['총매물수']) * 100
        min_cnt = 5
        all_dealer = dealer_counts[dealer_counts['총매물수'] >= min_cnt]
        high_D_dealer = all_dealer[all_dealer['D등급비율(%)'] > 80]

        num_all = len(all_dealer)
        num_high = len(high_D_dealer)
        ratio = (num_high / num_all * 100) if num_all > 0 else 0

        st.subheader("D(매우 주의) 매물 비중 80% 초과 딜러 비율")
        st.markdown(f"""
        **판매중 매물의 80% 이상이 D(매우 주의) 등급인 딜러는 전체 딜러 중 <span style="color:#ec5656;font-weight:bold;">{ratio:.1f}%</span> ({num_high}명 / {num_all}명)입니다.**
        """, unsafe_allow_html=True)
        fig2 = px.pie(
            names=['D80% 초과', '그외'],
            values=[num_high, max(num_all - num_high, 0)],
            color=['D80% 초과', '그외'],
            color_discrete_map={
                'D80% 초과': "#ec5656",
                '그외': "#bbbbbb"
            },
            hole=0.52
        )
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)


        st.subheader("D등급 편중 딜러가 차지하는 비중")
        all_D_vehicles = set(df[df['등급'] == 'D (매우 주의)']['vehicleNo'])
        high_D_dealer_vehicles = set(
            df[(df['등급'] == 'D (매우 주의)') & (df['딜러식별'].isin(high_D_dealer['딜러식별']))]['vehicleNo']
        )
        n_all_D = len(all_D_vehicles)
        n_high_D = len(high_D_dealer_vehicles)
        D_vehicles_ratio = (n_high_D / n_all_D * 100) if n_all_D > 0 else 0

        st.markdown(f"""
        **D등급 매물 비중이 80% 넘는 딜러가 판매하는 D(매우 주의) 매물 개수는 <span style="color:#ec5656;font-weight:bold;">{n_high_D:,}대</span>이고,   
        이는 전체 D매물의 <span style="color:#ec5656;font-weight:bold;">{D_vehicles_ratio:.1f}%</span>에 해당합니다.**
        """, unsafe_allow_html=True)

        fig3 = px.pie(
            names=['D80% 초과 딜러 D매물', '기타 딜러 D매물'],
            values=[n_high_D, max(n_all_D - n_high_D, 0)],
            color=['D80% 초과 딜러 D매물', '기타 딜러 D매물'],
            color_discrete_map={
                'D80% 초과 딜러 D매물': "#ec5656",
                '기타 딜러 D매물': "#bbbbbb"
            },
            hole=0.5
        )
        fig3.update_traces(textinfo='percent+label')
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("※ D(매우 주의) 편중 딜러들의 D매물 집중도를 시각적으로 보여줍니다. (매물 5건 이상 딜러만 포함)")

    if __name__ == "__main__":
        main()
