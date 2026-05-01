import os
import sys
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setup_rag import get_relevant_sop


# ── 模擬工具函數（Agentic Tool Use / MCP 模擬）──────────────────────────

def check_inventory(part_name: str) -> dict:
    """查詢指定零件的庫存狀態與位置。

    Args:
        part_name: 零件名稱，例如「射頻模組」或「光纖」。

    Returns:
        包含 status（庫存狀態）、quantity（數量）、location（位置）的字典。
    """
    if any(k in part_name for k in ["射頻模組", "光纖"]):
        return {"status": "充足", "quantity": 3, "location": "信義區機房"}
    return {"status": "庫存不足", "quantity": 0, "location": "N/A"}


def check_engineer_schedule(area: str) -> dict:
    """查詢指定區域的外勤工程師班表與預計到達時間。

    Args:
        area: 區域名稱，例如「信義區」。

    Returns:
        包含 engineer（工程師姓名）、status（待命狀態）、eta（預計到達時間）的字典。
    """
    if "信義" in area:
        return {"engineer": "王大明", "status": "待命中", "eta": "15分鐘"}
    return {"engineer": "N/A", "status": "不在班", "eta": "未知"}


# ── 預測趨勢函數 ──────────────────────────────────────────────────────────

def predict_trend(values: list, n_steps: int = 5) -> list:
    """使用一元線性迴歸（polyfit degree=1）預測未來 n_steps 步的數值。"""
    if len(values) < 2:
        return [float(values[-1])] * n_steps
    x = np.arange(len(values), dtype=float)
    coeffs = np.polyfit(x, values, 1)
    future_x = np.arange(len(values), len(values) + n_steps, dtype=float)
    return np.polyval(coeffs, future_x).tolist()


# ── Streamlit 設定 ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="中華電信行動維運 AI 系統",
    page_icon=None,
    layout="wide",
)

st.title("中華電信行動維運 - 跨年告警與自動派工 AI 系統 (PoC)")
st.caption("信義區跨年晚會 | 2023-12-31 23:45 ~ 2024-01-01 00:15")

# --- 讀取 API Key ---
api_key = os.environ.get("GEMINI_API_KEY", "")


@st.cache_data
def load_data():
    df = pd.read_csv("network_mock_data.csv", encoding="utf-8-sig")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


df = load_data()
timestamps = df["Timestamp"].tolist()

# --- 側邊欄 ---
with st.sidebar:
    st.header("控制面板")
    st.markdown("---")
    st.subheader("時間軸選擇")

    selected_index = st.slider(
        "當前時間 (Timestamp)",
        min_value=0,
        max_value=len(timestamps) - 1,
        value=len(timestamps) - 1,
        format="",
        help="拖曳以模擬系統在不同時間點的狀態",
    )

    selected_ts = timestamps[selected_index]

    if selected_ts.day == 1:
        display_time = "00:" + selected_ts.strftime("%M")
    else:
        display_time = selected_ts.strftime("%H:%M")

    st.metric(label="選定時間點", value=display_time)
    st.markdown("---")

    current_row = df[df["Timestamp"] == selected_ts].iloc[0]
    st.subheader("當前指標快照")
    st.metric("PRB 利用率", f"{current_row['PRB_Utilization']:.1f}%")
    st.metric("RRC 建立成功率", f"{current_row['RRC_Setup_Success_Rate']:.2f}%")
    st.metric("換手失敗率", f"{current_row['Handover_Failure_Rate']:.2f}%")

# 篩選歷史數據
df_history = df[df["Timestamp"] <= selected_ts].copy()

# ── 預測性維運計算 ────────────────────────────────────────────────────────
PREDICT_STEPS = 5
tail5 = df_history.tail(5)

if len(df_history) >= 2:
    time_delta = df_history["Timestamp"].iloc[-1] - df_history["Timestamp"].iloc[-2]
else:
    time_delta = pd.Timedelta(minutes=5)

future_ts = [selected_ts + time_delta * (i + 1) for i in range(PREDICT_STEPS)]

prb_pred = [min(max(v, 0.0), 105.0) for v in predict_trend(tail5["PRB_Utilization"].tolist())]
rrc_pred = [min(max(v, 0.0), 105.0) for v in predict_trend(tail5["RRC_Setup_Success_Rate"].tolist())]

prb_pred_final = prb_pred[-1]
rrc_pred_final = rrc_pred[-1]
prb_trend_dir = "上升" if prb_pred_final > prb_pred[0] else ("下降" if prb_pred_final < prb_pred[0] else "平穩")
rrc_trend_dir = "下降" if rrc_pred_final < rrc_pred[0] else ("上升" if rrc_pred_final > rrc_pred[0] else "平穩")
prediction_summary = (
    f"PRB 趨勢：{prb_trend_dir}，預測 {PREDICT_STEPS} 分鐘後達 {prb_pred_final:.1f}%；"
    f"RRC 趨勢：{rrc_trend_dir}，預測 {PREDICT_STEPS} 分鐘後達 {rrc_pred_final:.1f}%"
)

# --- 戰情儀表板 ---
st.subheader("戰情儀表板")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("PRB 利用率 (%)", "RRC 連線建立成功率 (%)"),
    horizontal_spacing=0.08,
)

# 實際值
fig.add_trace(
    go.Scatter(
        x=df_history["Timestamp"],
        y=df_history["PRB_Utilization"],
        mode="lines+markers",
        name="PRB 實際值",
        line=dict(color="#FF6B35", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.12)",
    ),
    row=1, col=1,
)
fig.add_hline(y=85, line_dash="dash", line_color="red", opacity=0.6,
              annotation_text="警戒值 85%", annotation_position="top left", row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=df_history["Timestamp"],
        y=df_history["RRC_Setup_Success_Rate"],
        mode="lines+markers",
        name="RRC 實際值",
        line=dict(color="#00B4D8", width=2),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.12)",
    ),
    row=1, col=2,
)
fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.6,
              annotation_text="警戒值 80%", annotation_position="bottom left", row=1, col=2)

# 預測值（虛線）
fig.add_trace(
    go.Scatter(
        x=future_ts,
        y=prb_pred,
        mode="lines",
        name="PRB 預測",
        line=dict(color="#FFB347", width=2, dash="dash"),
        opacity=0.85,
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=future_ts,
        y=rrc_pred,
        mode="lines",
        name="RRC 預測",
        line=dict(color="#7DF9FF", width=2, dash="dash"),
        opacity=0.85,
    ),
    row=1, col=2,
)

fig.update_yaxes(range=[0, 105], row=1, col=1)
fig.update_yaxes(range=[40, 105], row=1, col=2)
fig.update_layout(
    height=360,
    showlegend=True,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=10),
    ),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="#FAFAFA"),
    margin=dict(l=40, r=40, t=50, b=40),
)
fig.update_xaxes(gridcolor="#2C2F36", showgrid=True, tickformat="%H:%M")
fig.update_yaxes(gridcolor="#2C2F36", showgrid=True)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

col_alert, col_ai = st.columns([1, 1], gap="large")

# --- 即時告警區 ---
with col_alert:
    st.subheader("即時告警區")

    log_msg = current_row["System_Log"]
    is_critical = any(keyword in log_msg for keyword in ["Alarm", "Failure"])

    if is_critical:
        st.markdown(
            f"""
            <div style="background-color:#2D1B1B;border:1px solid #FF4B4B;
                        border-radius:8px;padding:16px;">
                <span style="color:#FF4B4B;font-size:18px;font-weight:bold;">
                    {log_msg}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#1B2D1B;border:1px solid #21C55D;
                        border-radius:8px;padding:16px;">
                <span style="color:#21C55D;font-size:18px;font-weight:bold;">
                    {log_msg}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("最近 5 筆告警歷史")

    recent_logs = df_history.tail(5)[["Timestamp", "System_Log"]].copy()
    recent_logs["Timestamp"] = recent_logs["Timestamp"].dt.strftime("%H:%M")
    recent_logs = recent_logs.rename(columns={"Timestamp": "時間", "System_Log": "系統訊息"})

    def highlight_alarm(row):
        if any(k in row["系統訊息"] for k in ["Alarm", "Failure"]):
            return ["color: #FF4B4B"] * len(row)
        return ["color: #21C55D"] * len(row)

    st.dataframe(
        recent_logs.style.apply(highlight_alarm, axis=1),
        use_container_width=True,
        hide_index=True,
    )

# --- AI 維運助理區 ---
with col_ai:
    st.subheader("AI 維運助理")

    st.markdown(
        """
        <div style="background-color:#1A1F2E;border:1px solid #3A4060;
                    border-radius:8px;padding:16px;margin-bottom:12px;">
            <p style="color:#8892B0;margin:0;">
                點擊下方按鈕，AI 助理將根據即時告警數據、預測趨勢與內部 SOP，
                自動進行根因分析、工具調用，並產出五段完整處置報告。
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    if st.button("啟動 AI 智能診斷", type="primary", use_container_width=True):

        if not api_key:
            st.error(
                "未偵測到 GEMINI_API_KEY。\n\n"
                "請在 .env 檔案中設定：\n"
                "`GEMINI_API_KEY=AIzaSy...`"
            )
        else:
            try:
                status = st.status("AI 助理正在進行診斷與規劃...", expanded=True)

                # [步驟 A] 擷取當前 KPI 狀態
                prb   = current_row["PRB_Utilization"]
                rrc   = current_row["RRC_Setup_Success_Rate"]
                ho    = current_row["Handover_Failure_Rate"]
                alarm = current_row["System_Log"]

                kpi_summary = (
                    f"時間：{display_time}｜"
                    f"PRB 利用率：{prb:.1f}%｜"
                    f"RRC 建立成功率：{rrc:.2f}%｜"
                    f"換手失敗率：{ho:.2f}%｜"
                    f"系統告警：{alarm}"
                )

                # [步驟 B] RAG 檢索最相關 SOP
                rag_query = (
                    f"當前網路狀態：PRB {prb:.1f}%，"
                    f"RRC 成功率 {rrc:.2f}%，"
                    f"換手失敗率 {ho:.2f}%，"
                    f"告警訊息：{alarm}"
                )
                sop_hits = get_relevant_sop(rag_query, n_results=1)
                sop_content = sop_hits[0]["document"] if sop_hits else "無可用 SOP"
                sop_id      = sop_hits[0]["id"]       if sop_hits else "N/A"

                # [步驟 C] 建構 Prompt（含預測趨勢注入）
                user_prompt = (
                    f"【告警數據】{kpi_summary}\n\n"
                    f"【未來 {PREDICT_STEPS} 分鐘預測趨勢】{prediction_summary}\n\n"
                    f"【SOP 參考】({sop_id}) {sop_content}\n\n"
                    f"請輸出五個段落：\n"
                    f"異常狀態研判 / 未來趨勢預測 / 根因分析 / "
                    f"建議參數調整 / 派工與物料調度（需包含工具查詢結果）"
                )

                system_instruction = (
                    "你是中華電信資深網路維運 AI 代理人，根據數據、趨勢預測與 SOP 進行根因分析，"
                    "用 Markdown 格式回答，保持專業簡潔。"
                    "若判斷為硬體故障，你必須先呼叫 check_inventory 查詢零件庫存，"
                    "再呼叫 check_engineer_schedule 確認工程師班表，最後才產出完整報告。"
                    "報告必須包含五個段落：異常狀態研判、未來趨勢預測、根因分析、"
                    "建議參數調整、派工與物料調度（需包含工具查詢結果）。"
                )

                # [步驟 D] Agentic Function Calling 迴圈
                client = genai.Client(api_key=api_key)
                config = genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=[check_inventory, check_engineer_schedule],
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                    max_output_tokens=2000,
                    temperature=0.7,
                )

                conversation = [genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=user_prompt)],
                )]

                response = None
                for _iter in range(5):
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=conversation,
                            config=config,
                        )
                    except Exception as api_err:
                        err_str = str(api_err)
                        if "429" in err_str:
                            st.warning(
                                "偵測到 API 配額限制 (429)，請稍候 30 秒再試，"
                                "或檢查 Google AI Studio 是否已開啟該型號的 Free Tier 權限。"
                            )
                        else:
                            st.error(f"診斷過程發生錯誤：{api_err}")
                        st.stop()

                    model_content = response.candidates[0].content
                    conversation.append(model_content)

                    # 找出所有 function_call parts
                    fn_calls = [
                        p for p in model_content.parts
                        if p.function_call is not None
                    ]
                    if not fn_calls:
                        break  # 無工具調用，最終回答已就緒

                    # 執行工具並收集回應
                    fn_response_parts = []
                    for part in fn_calls:
                        fc = part.function_call
                        fn_args = dict(fc.args)
                        if fc.name == "check_inventory":
                            status.write("正在連線後台查詢零件庫存...")
                            result = check_inventory(**fn_args)
                            status.write("庫存查詢完成！")
                        elif fc.name == "check_engineer_schedule":
                            status.write("正在確認外勤工程師排班系統...")
                            result = check_engineer_schedule(**fn_args)
                            status.write("排班查詢完成！")
                        else:
                            result = {"error": f"Unknown tool: {fc.name}"}

                        fn_response_parts.append(
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=fc.name,
                                    response=result,
                                )
                            )
                        )

                    conversation.append(genai_types.Content(
                        role="user",
                        parts=fn_response_parts,
                    ))
                    final_reminder = (
                        "請結合上述的即時告警數據、SOP、未來預測趨勢，以及剛剛取得的工具查詢結果，"
                        "務必完整產出包含以下五個段落的 Markdown 報告，缺一不可：\n"
                        "- 🚨 異常狀態研判\n"
                        "- 📈 未來趨勢預測\n"
                        "- 🔍 根因分析推測\n"
                        "- ⚙️ 系統建議參數調整\n"
                        "- 🛠️ 外勤派工與物料調度（請帶入庫存與排班查詢結果）"
                    )
                    conversation.append(genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=final_reminder)],
                    ))

                full_response = response.text if response else "（無回應）"
                status.update(label="AI 診斷與調度完成", state="complete", expanded=False)

                # [步驟 E] 渲染結果（先佔位再填入，避免介面閃爍）
                result_placeholder = st.empty()
                with result_placeholder.container():
                    st.markdown(
                        f"""
                        <div style="background-color:#0F172A;border:1px solid #3B82F6;
                                    border-radius:8px;padding:4px 16px 4px 16px;
                                    margin-top:8px;">
                            <p style="color:#60A5FA;font-size:11px;margin:8px 0 4px 0;">
                                RAG 命中：{sop_id} ｜ 模型：gemini-2.5-flash ｜ 時間點：{display_time}
                            </p>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    st.markdown(full_response)

            except Exception as e:
                st.error(f"診斷過程發生錯誤：{e}")
